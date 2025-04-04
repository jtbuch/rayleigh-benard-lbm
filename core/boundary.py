from __future__ import annotations

import abc
from collections.abc import Iterable
from typing import TYPE_CHECKING, Dict, Union, Sequence, Any

from multipledispatch import dispatch

import chex
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class

from .lattice import Lattice, CoupledLattices

if TYPE_CHECKING:
    from .simulation import LBMState

__all__ = [
    "BounceBackBoundary",
    "InletBoundary",
    "OutletBoundary",
    "BoundaryDict",
    "apply_boundary_conditions",
]

class Boundary(abc.ABC):
    _name: str
    _mask: chex.Array
    _size: int

    def __init__(self, name: str, mask: chex.Array, _unflatten: bool=False, _size: int=None, _mask_id=None) -> None:
        self._name = name
        self._mask = mask
        if _unflatten:
            self._size = _size
            self._mask_id = _mask_id
        else:
            self._size = jnp.sum(mask, dtype=jnp.int32)
            self._mask_id = jnp.where(self._mask, size=self._size)

    def set_param(self, key: str, val: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, val)
        else:
            raise ValueError(
                f"Attempted to update unknown parameter '{key}' of boundary '{self._name}'."
            )

    @abc.abstractmethod
    def __call__(self, lattice: Lattice, df: chex.Array, *args, **kwargs) -> chex.Array:
        """Apply the boundary condition to the given distribution function.

        Args:
            lattice (Lattice): The lattice.
            df (chex.Array): The distribution function.

        Returns:
            chex.Array: The distribution function after the application of the
                boundary condition.
        """
        pass

    @property
    def name(self):
        return self._name

    @property
    @abc.abstractmethod
    def collision_mask(self):
        """
        Return the mask of fluid nodes on which the collision step is performed.
        """
        pass

    @property
    @abc.abstractmethod
    def stream_mask(self):
        """
        Return the mask of fluid nodes on which the streaming step is performed.
        """
        pass

    def tree_flatten(self):
        # arrays / dynamic values
        children = (self._mask, self._mask_id)
        # static values
        aux_data = {
            "name": self._name,
            "_size": self._size,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        unflattened = cls(
            mask=children[0],
            name=aux_data["name"],
            _size=aux_data["_size"],
            _mask_id=children[1],
            _unflatten=True
        )
        return unflattened

@register_pytree_node_class
class BoundaryDict:
    _boundary_dict: Dict[str, Boundary]
    _collision_mask: chex.Array = True
    _stream_mask: chex.Array = True

    def __init__(
        self,
        boundary: Union[Boundary, Sequence[Boundary]] = None,
    ):
        self._boundary_dict = dict()
        if boundary is not None:
            self.add(boundary)

    def set_params(self, boundary_name: str, value: Dict) -> None:
        """Set the parameters of a boundary.

        Args:
            boundary_name (str): The name of the boundary.
            value (Dict): The parameters of the boundary.
        """
        if boundary_name not in self._boundary_dict:
            raise ValueError(f"Boundary '{boundary_name}' does not exist.")

        for key, val in value.items():
            self._boundary_dict[boundary_name].set_param(key, val)

    @dispatch(Boundary)
    def add(self, boundary: Boundary) -> None:
        """Add a boundary to the list of boundaries.

        Args:
            boundary (Union[Boundary, Sequence[Boundary]]): The boundaries to add.
        """
        if boundary.name in self._boundary_dict:
            raise ValueError(f"Boundary {boundary.name} already exists.")
        self._boundary_dict[boundary.name] = boundary
        self.update_masks()

    @dispatch(Iterable)
    def add(self, boundary: Sequence[Boundary]) -> None:
        """Add a boundary to the list of boundaries.

        Args:
            boundary (Sequence[Boundary]): The boundaries to add.
        """
        for bdy in boundary:
            self.add(bdy)

    def __call__(self, lattice: Lattice, df: chex.Array) -> chex.Array:
        """Apply all the boundary conditions to the given distribution function.

        Args:
            lattice (Lattice): The lattice.
            df (chex.Array): The distribution function.

        Returns:
            Array: The distribution function after the application of the boundary
                conditions.
        """
        for name, bdy in self._boundary_dict.items():
            df = bdy(lattice, df)
        return df

    def update_masks(self) -> None:
        masks = jnp.array(
            [bdy.collision_mask for bdy in self._boundary_dict.values()], dtype=bool
        )
        self._collision_mask = jnp.prod(masks, dtype=bool, axis=0)[..., jnp.newaxis]

        masks = jnp.array(
            [b.stream_mask for b in self._boundary_dict.values()], dtype=bool
        )
        self._stream_mask = jnp.prod(masks, dtype=bool, axis=0)

    @property
    def collision_mask(self) -> chex.Array:
        return self._collision_mask

    @property
    def stream_mask(self) -> chex.Array:
        return self._stream_mask

    def tree_flatten(self):
        children = (self._boundary_dict, self.stream_mask, self.collision_mask)
        return (children, {})

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        unflattened = cls()
        unflattened._boundary_dict = children[0]
        unflattened._stream_mask = children[1]
        unflattened._collision_mask = children[2]
        return unflattened


@register_pytree_node_class
class BounceBackBoundary(Boundary):
    """A bounce back boundary condition. The distribution function is set to the
    opposite distribution function on the boundary nodes. It can be used to model a
    solid wall.
    """

    def __call__(self, lattice: Lattice, df: chex.Array) -> chex.Array:
        """Apply the bounce back boundary condition to the given distribution function.

        Args:
            lattice (Lattice): The lattice.
            df (chex.Array): The distribution function.

        Returns:
            Array: The distribution function after the application of the boundary
                condition.
        """
        idx, idy = self._mask_id
        bounced = df[:,:,lattice.stencil.opposite]
        df = df.at[idx,idy,:].set(bounced[idx, idy, :])
        return df

    @property
    def collision_mask(self):
        # Collision is performed on all nodes except the boundary nodes.
        return ~self._mask

    @property
    def stream_mask(self):
        # Streaming is performed on all nodes except the boundary nodes.
        return ~self._mask

@register_pytree_node_class
class InletBoundary(Boundary):
    """An inlet boundary condition. The distribution function is set to the
    equilibrium distribution function on the boundary nodes. It can be used to model
    an inlet.
    """
    m = 1.0
    u = jnp.array([0.0, 0.0])

    def __call__(
        self,
        lattice: Lattice,
        df: chex.Array,
    ) -> chex.Array:
        """Apply the inlet boundary condition to the given distribution function.
        Args:
            lattice (Lattice): The lattice.
            df (chex.Array): The distribution function.

        Returns:
            Array: The distribution function after the application of the boundary
                condition.
        """
        m = self.m
        u = self.u
        if isinstance(m, chex.Scalar):
            m = m * jnp.ones((self._size))
        m = m[..., jnp.newaxis]
        
        if u.ndim == 1:
            u = jnp.ones((self._size, lattice.D)) * u[jnp.newaxis, :]
        
        eq = lattice.equilibrium(m, u)
        idx, idy = self._mask_id
        df = df.at[idx,idy,:].set(eq)
        return df

    @property
    def collision_mask(self):
        # Collision is performed on all nodes
        return jnp.ones_like(~self._mask)

    @property
    def stream_mask(self):
        # Streaming is performed on all nodes except the boundary nodes.
        return ~self._mask

    def tree_flatten(self):
        # arrays / dynamic values
        children = (self._mask, self._mask_id, self.m, self.u)
        # static values
        aux_data = {
            "name": self._name,
            "_size": self._size,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        unflattened = cls(
            mask=children[0],
            name=aux_data["name"],
            _size=aux_data["_size"],
            _mask_id=children[1],
            _unflatten=True
        )
        unflattened.m = children[2]
        unflattened.u = children[3]
        return unflattened

@register_pytree_node_class
class OutletBoundary(Boundary):
    """An outlet boundary condition."""

    def __init__(self, name, mask: chex.Array, direction: Sequence[int], _unflatten: bool=False, _neigh_idx=None, _size: int=None, _mask_id=None) -> None:
        if _unflatten:
            self._neigh_idx = _neigh_idx
        else:
            direction = jnp.array(direction, dtype=jnp.int8)
            self._neigh_idx = jnp.roll(mask, shift=-direction, axis=(0, 1))
            self._neigh_idx = jnp.where(self._neigh_idx)
        super().__init__(name, mask, _unflatten, _size, _mask_id)

    def __call__(self, lattice: Lattice, df: chex.Array) -> chex.Array:
        """Apply the outlet boundary condition to the given distribution function.
        Args:
            lattice (Lattice): The lattice.
            df (chex.Array): The distribution function.

        Returns:
            Array: The distribution function after the application of the boundary
                condition.
        """
        # Equilibrium on neighbor lattice points
        idx_n, idy_n = self._neigh_idx
        fluid_state = lattice.get_macroscopics(df[idx_n, idy_n,:])
        eq = lattice.equilibrium(fluid_state)

        idx, idy = self._mask_id
        df = df.at[idx,idy,:].set(eq)
        return df

    @property
    def collision_mask(self):
        # Collision is performed on all nodes.
        return ~self._mask

    @property
    def stream_mask(self):
        # Streaming is performed on all nodes except the boundary nodes.
        return ~self._mask

    def tree_flatten(self):
        # arrays / dynamic values
        children = (self._mask, self._mask_id, self._neigh_idx)
        # static values
        aux_data = {
            "name": self._name,
            "_size": self._size,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        unflattened = cls(
            name=aux_data["name"],
            mask=children[0],
            direction=None,
            _unflatten=True,
            _size=aux_data["_size"],
            _mask_id=children[1],
            _neigh_idx=children[2],
        )
        return unflattened

def apply_boundary_conditions(
    lattice: Union[Lattice, CoupledLattices],
    state_dict: Dict[str, LBMState],
) -> Dict[str, LBMState]:
    """Apply all the boundary conditions to the given distribution function.

    Args:
        lattice (Union[Lattice, CoupledLattices],): The lattice.
        state_dict (Dict[str, LBMState]): The state dictionary, containing the
            distribution function and the boundary conditions.

    Returns:
        (Dict[str, LBMState]): The state dictionary updated according to the boundary
            conditions.
    """
    for name in state_dict.keys():
        if isinstance(lattice, CoupledLattices):
            l = lattice[name]
        else:
            l = lattice

        state_dict[name].df = state_dict[name].bc(l, state_dict[name].df)

    return state_dict

