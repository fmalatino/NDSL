from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import xarray as xr

from ndsl.comm.comm_abc import Comm as CommABC
from ndsl.comm.communicator import Communicator
from ndsl.comm.mpi import get_mpi_type


if TYPE_CHECKING:
    from _typeshed import DataclassInstance

T = TypeVar("T", bound="DataclassInstance")


def _shape_list_gen(
    var_shape: tuple,
    pelist_size: int,
    locale: str,
    nhalo: int = 0,
    i_adj: int = 0,
    j_adj: int = 0,
    k_adj: int = 0,
) -> list:
    shape_list = []
    for n in range(pelist_size):
        match locale:
            case "right" | "left":
                i = var_shape[0] - i_adj
                j = nhalo
            case "top" | "bottom":
                i = nhalo
                j = (
                    var_shape[1] - j_adj - nhalo
                    if n in (0, pelist_size - 1)
                    else var_shape[1] - j_adj
                )
        if len(var_shape) == 3:
            k = var_shape[2] - k_adj
            shape_list.append((i,j,k))
        else:
            shape_list.append((i,j))
    return shape_list

def _get_displs(counts: list) -> list:
    displs = [0]
    for n in range(1, len(counts)):
        displs.append(displs[n - 1] + counts[n - 1])
    return displs

class BoundaryCondition:
    file: Path
    dataset: xr.Dataset
    var_list: list[str]
    _location: str
    _color: int
    _main_comm: Communicator
    _sub_com: CommABC
    _sub_com_rank: int
    _sub_com_size: int

    def __init__(
        self,
        comm: Communicator,
        layout: tuple,
        locale: str,
        file: Path = None,
    ):
        self.file = Path(file)
        self._main_comm = comm
        pelist = []
        self._location = locale.lower()
        match self._location:
            case "top":
                for n in range(layout[1]):
                    pelist.append(layout[1] * (layout[1] - 1) + n)
                self._color = 1 if comm.rank in pelist else 0
            case "bottom":
                for n in range(layout[1]):
                    pelist.append(n)
                self._color = 1 if comm.rank in pelist else 0
            case "right":
                for n in range(layout[0]):
                    pelist.append(layout[0] * (n + 1) - 1)
                self._color = 1 if comm.rank in pelist else 0
            case "left":
                for n in range(layout[0]):
                    pelist.append(layout[0] * n)
                self._color = 1 if comm.rank in pelist else 0
            case _:
                raise ValueError(f"{locale} is not an edge position")
        self._sub_com = self._main_comm.comm.Split(
            color=self._color, key=self._main_comm.rank
        )
        self._sub_com_rank = self._sub_com.Get_rank()
        self._sub_com_size = self._sub_com.Get_size()
        self.var_list = []
        if self._sub_com_rank == 0 and self._color == 1:
            if self.file.is_file():
                self.dataset = xr.open_dataset(file)
                whole_var_list = list(self.dataset.keys())
                self.var_list = [
                    var for var in whole_var_list if self._location in var.lower()
                ]
            else:
                self.dataset = None
        self.var_list = self._sub_com.bcast(self.var_list) or []

    def scatter_bcs(self, state: T, timestep: int) -> None:
        if self._color == 1:
            for field_obj in dataclasses.fields(state):
                var_name = field_obj.name
                if var_name in self.var_list:
                    var = getattr(state, var_name)
                    var_shape = var.shape
                    n_halo = var.metadata.n_halo
                    iadj = 1 if var.dims[0] == "i" else 0
                    jadj = 1 if var.dims[1] == "j" else 0
                    kadj = 1 if (len(var.dims) == 3 and var.dims[2] == "k") else 0
                    shape_list = _shape_list_gen(
                        var_shape=var_shape,
                        pelist_size=self._sub_com_size,
                        locale=self._location,
                        nhalo=n_halo,
                        i_adj=iadj,
                        j_adj=jadj,
                        k_adj=kadj,
                    )
                    recv_buf = np.empty(
                        shape=shape_list[self._sub_com_rank], dtype=var.dtype
                    ).flatten()
                    if self._sub_com_rank == 0:
                        da = np.ascontiguousarray(self.dataset[var_name].data).flatten()
                        sendcounts = [
                            np.prod(shape_list[n]) for n in range(self._sub_com_size)
                        ]
                        displs = _get_displs(sendcounts)
                        temp = np.empty(shape=sum(sendcounts), dtype=da.dtype)
                        datatype = get_mpi_type(da)
                    else:
                        temp = None
                        sendcounts = None
                        displs = None
                        datatype = None
                    if self._sub_com_rank == 0:
                        m = 0
                        assert sendcounts is not None
                        for n in range(self._sub_com_size):
                            temp[m : m + sendcounts[n]] = da[m : m + sendcounts[n]]
                            m += sendcounts[n]
                    self._sub_com.Scatterv(
                        [temp, sendcounts, displs, datatype], recv_buf, root=0
                    )
                    match self._location:
                        case "top":
                            js = 0
                            je = shape_list[self._sub_com_rank][1]
                            if self._sub_com_rank == 0:
                                js = n_halo
                                je = shape_list[self._sub_com_rank][1] + n_halo
                            if len(var_shape) == 2:
                                var[:n_halo, js:je] = recv_buf[:].reshape(shape_list[self._sub_com_rank])
                            if len(var_shape) == 3:
                                var[:n_halo, js:je, : var_shape[2] - kadj] = recv_buf[:].reshape(shape_list[self._sub_com_rank])
                        case "bottom":
                            js = 0
                            je = shape_list[self._sub_com_rank][1]
                            if self._sub_com_rank == 0:
                                js = n_halo
                                je = shape_list[self._sub_com_rank][1] + n_halo
                            if len(var_shape) == 2:
                                var[
                                    var_shape[0] - n_halo - iadj : var_shape[0] - iadj,
                                    js:je
                                ] = recv_buf[:].reshape(shape_list[self._sub_com_rank])
                            if len(var_shape) == 3:
                                var[
                                    var_shape[0] - n_halo - iadj : var_shape[0] - iadj,
                                    js:je,
                                    : var_shape[2] - kadj,
                                ] = recv_buf[:].reshape(shape_list[self._sub_com_rank])
                        case "right":
                            if len(var_shape) == 2:
                                var[
                                    : var_shape[0] - iadj,
                                    var_shape[1] - n_halo - jadj : var_shape[1] - jadj,
                                ] = recv_buf[:].reshape(shape_list[self._sub_com_rank])
                            if len(var_shape) == 3:
                                var[
                                    : var_shape[0] - iadj,
                                    var_shape[1] - n_halo - jadj : var_shape[1] - jadj,
                                    : var_shape[2] - kadj,
                                ] = recv_buf[:].reshape(shape_list[self._sub_com_rank])  
                        case "left":
                            if len(var_shape) == 2:
                                var[
                                    : var_shape[0] - iadj,
                                    :n_halo,
                                ] = recv_buf[:].reshape(shape_list[self._sub_com_rank])
                            if len(var_shape) == 3:
                                var[
                                    : var_shape[0] - iadj,
                                    :n_halo,
                                    : var_shape[2] - kadj,
                                ] = recv_buf[:].reshape(shape_list[self._sub_com_rank])
                    setattr(state, var_name, var)

    def write_out_bcs(
        self,
        state: T,
        bc_file_name: Path = None,
    ) -> None:
        if self._color == 1:
            if bc_file_name is None:
                bc_file_name = self.file
            for field_obj in dataclasses.fields(state):
                var_name = field_obj.name
                if self._location in var_name.lower():
                    if var_name not in self.var_list:
                        self.var_list.append(var_name)
                    var = getattr(state, var_name)
                    var_shape = var.shape
                    n_halo = var.metadata.n_halo
                    iadj = 1 if var.dims[0] == "i" else 0
                    jadj = 1 if var.dims[1] == "j" else 0
                    kadj = 1 if (len(var.dims) == 3 and var.dims[2] == "k") else 0
                    shape_list = _shape_list_gen(
                        var_shape=var_shape,
                        pelist_size=self._sub_com_size,
                        locale=self._location,
                        nhalo=n_halo,
                        i_adj=iadj,
                        j_adj=jadj,
                        k_adj=kadj,
                    )
                    send_buf = np.empty(
                        shape=shape_list[self._sub_com_rank], dtype=var.dtype
                    ).flatten()
                    match self._location:
                        case "top":
                            js = 0
                            je = shape_list[self._sub_com_rank][1]
                            if self._sub_com_rank == 0:
                                js = n_halo
                                je = shape_list[self._sub_com_rank][1] + n_halo
                            if len(var_shape) == 2:
                                send_buf[:] = var[:n_halo, js:je].flatten()
                            if len(var_shape) == 3:
                                send_buf[:] = var[:n_halo, js:je, : var_shape[2] - kadj].flatten()
                        case "bottom":
                            js = 0
                            je = shape_list[self._sub_com_rank][1]
                            if self._sub_com_rank == 0:
                                js = n_halo
                                je = shape_list[self._sub_com_rank][1] + n_halo
                            if len(var_shape) == 2:
                                send_buf[:] = var[
                                    var_shape[0] - n_halo - iadj : var_shape[0] - iadj,
                                    js:je,
                                ].flatten()
                            if len(var_shape) == 3:
                                send_buf[:] = var[
                                    var_shape[0] - n_halo - iadj : var_shape[0] - iadj,
                                    js:je,
                                    : var_shape[2] - kadj,
                                ].flatten()
                        case "right":
                            if len(var_shape) == 2:
                                send_buf[:] = var[
                                    : var_shape[0] - iadj,
                                    var_shape[1] - n_halo - jadj : var_shape[1] - jadj,
                                ].flatten()
                            if len(var_shape) == 3:
                                send_buf[:] = var[
                                    : var_shape[0] - iadj,
                                    var_shape[1] - n_halo - jadj : var_shape[1] - jadj,
                                    : var_shape[2] - kadj,
                                ].flatten()
                        case "left":
                            if len(var_shape) == 2:
                                send_buf[:] = var[
                                    : var_shape[0] - iadj,
                                    :n_halo,
                                ].flatten()
                            if len(var_shape) == 3:
                                send_buf[:] = var[
                                    : var_shape[0] - iadj,
                                    :n_halo,
                                    : var_shape[2] - kadj,
                                ].flatten()
                    if self._sub_com_rank == 0:
                        sendcounts = [
                            np.prod(shape_list[n]) for n in range(self._sub_com_size)
                        ]
                        displs = _get_displs(sendcounts)
                        temp = np.empty(shape=sum(sendcounts), dtype=var.dtype)
                        datatype = get_mpi_type(var[:])
                    else:
                        sendcounts = None
                        displs = None
                        temp = None
                        datatype = None
                    self._sub_com.Gatherv(send_buf, [temp, sendcounts, displs, datatype], root=0)
                    if self._sub_com_rank == 0:
                        if self.dataset == None:
                            self.dataset = xr.DataArray(temp, name=var_name).to_dataset()
                            self.dataset.to_netcdf(bc_file_name)
                        else:
                            self.dataset.assign(temp, name=var_name)
                            self.dataset.to_netcdf(bc_file_name, mode="w")

                            
