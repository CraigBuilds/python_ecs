from typing import List, Any, Callable, Iterable, Type, TypeVar, Tuple, get_args, Dict, Union
import inspect
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

class EntityId(int):
    """Wrapper around int to represent an entity id."""
    def __repr__(self):
        return f'EntityId({int(self)})'

class Empty(np.generic):
    """Placeholder for missing components."""
    pass

D = TypeVar('D', bound=np.generic, covariant=True)
def filtered(arr: npt.NDArray[Union[D, Empty]]) -> Iterable[D]:
    """Return an iterator over the array, filtering out Empty values. No copying is done."""
    return (x for x in arr if x is not Empty)

D1 = TypeVar('D1', bound=np.generic, covariant=True)
D2 = TypeVar('D2', bound=np.generic, covariant=True)
def filtered_zip(arr1: npt.NDArray[Union[D1, Empty]], arr2: npt.NDArray[Union[D2, Empty]]) -> Iterable[Tuple[D1, D2]]:
    """Return an iterator over the arrays, filtering out Empty values. No copying is done."""
    return ((x, y) for x, y in zip(arr1, arr2) if x is not Empty and y is not Empty)

D3 = TypeVar('D3', bound=np.generic, covariant=True)
def filtered_zip3(
    arr1: npt.NDArray[Union[D1, Empty]], arr2: npt.NDArray[Union[D2, Empty]], arr3: npt.NDArray[Union[D3, Empty]]
) -> Iterable[Tuple[D1, D2, D3]]:
    """Return an iterator over the arrays, filtering out Empty values. No copying is done."""
    return ((x, y, z) for x, y, z in zip(arr1, arr2, arr3) if x is not Empty and y is not Empty and z is not Empty)

D4 = TypeVar('D4', bound=np.generic, covariant=True)
def filtered_zip4(
    arr1: npt.NDArray[Union[D1, Empty]], arr2: npt.NDArray[Union[D2, Empty]], arr3: npt.NDArray[Union[D3, Empty]], arr4: npt.NDArray[Union[D4, Empty]]
) -> Iterable[Tuple[D1, D2, D3, D4]]:
    """Return an iterator over the arrays, filtering out Empty values. No copying is done."""
    return ((x, y, z, w) for x, y, z, w in zip(arr1, arr2, arr3, arr4) if x is not Empty and y is not Empty and z is not Empty and w is not Empty)

class Ctx:
    """
    Stores all entities and components in the game.
    Data is stored as a table, i.e., a set of NumPy arrays.
    Row index is the entity id, and columns store components.
    Entities and components are never removed.
    """

    def __init__(self, initial_capacity: int = 1024) -> None:
        self.entity_capacity: int = initial_capacity
        self.entity_count: int = 0  # Total number of entities
        self.components_by_type: Dict[Type, npt.NDArray[Union[Any, Empty]]] = {}

    def _expand_capacity(self) -> None:
        """Double the capacity of all component arrays to accommodate more entities."""
        old_capacity = self.entity_capacity
        self.entity_capacity *= 2
        for component_type, array in self.components_by_type.items():
            new_array = np.empty(self.entity_capacity, dtype=object)
            new_array[:old_capacity] = array
            new_array[old_capacity:] = Empty
            self.components_by_type[component_type] = new_array

    def add_entity(self) -> EntityId:
        """Add a new entity and return its EntityId."""
        entity_id = EntityId(self.entity_count)
        self.entity_count += 1
        if self.entity_count > self.entity_capacity:
            self._expand_capacity()
        return entity_id

    def add_component(self, entity: EntityId, component: Any) -> None:
        """Add a component to an entity."""
        assert 0 <= entity < self.entity_count, f"Entity {entity} does not exist."
        component_type = type(component)
        if component_type not in self.components_by_type:
            # Create a new component array for this type
            array = np.empty(self.entity_capacity, dtype=object)
            array[:] = Empty
            self.components_by_type[component_type] = array
        else:
            array = self.components_by_type[component_type]
        array[entity] = component

    def add_entity_with_components(self, *components: Any) -> EntityId:
        """Add a new entity with the given components."""
        entity_id = self.add_entity()
        for component in components:
            self.add_component(entity_id, component)
        return entity_id

    T = TypeVar('T')
    def make_unary_query(self, t: Type[T]) -> Iterable[T]:
        """Return an iterable of components of type t."""
        array = self.components_by_type.get(t)
        if array is None:
            return []
        else:
            return filtered(array)

    T1 = TypeVar('T1')
    T2 = TypeVar('T2')
    def make_binary_query(self, t1: Type[T1], t2: Type[T2]) -> Iterable[Tuple[T1, T2]]:
        """Return an iterable of tuples (Component1, Component2) for entities with both components."""
        arr1 = self.components_by_type.get(t1)
        arr2 = self.components_by_type.get(t2)
        if arr1 is None or arr2 is None:
            return []
        else:
            return filtered_zip(arr1, arr2)

    T3 = TypeVar('T3')
    def make_ternary_query(
        self, t1: Type[T1], t2: Type[T2], t3: Type[T3]
    ) -> Iterable[Tuple[T1, T2, T3]]:
        """Return an iterable of tuples (Component1, Component2, Component3) for entities with all three components."""
        arr1 = self.components_by_type.get(t1)
        arr2 = self.components_by_type.get(t2)
        arr3 = self.components_by_type.get(t3)
        if arr1 is None or arr2 is None or arr3 is None:
            return []
        else:
            return filtered_zip3(arr1, arr2, arr3)

    T4 = TypeVar('T4')
    def make_quaternary_query(
        self, t1: Type[T1], t2: Type[T2], t3: Type[T3], t4: Type[T4]
    ) -> Iterable[Tuple[T1, T2, T3, T4]]:
        """Return an iterable of tuples (Component1, Component2, Component3, Component4) for entities with all four components."""
        arr1 = self.components_by_type.get(t1)
        arr2 = self.components_by_type.get(t2)
        arr3 = self.components_by_type.get(t3)
        arr4 = self.components_by_type.get(t4)
        if arr1 is None or arr2 is None or arr3 is None or arr4 is None:
            return []
        else:
            return filtered_zip4(arr1, arr2, arr3, arr4)

Ty=TypeVar('Ty')
def infer_iterable_type_from_callable(f: Callable[[Iterable[Ty]], None]) -> Type[Ty]:
    """Infer the type T from a callable that takes an Iterable[T] as an argument."""
    func_signature = inspect.signature(f)
    params = list(func_signature.parameters.values())
    if not params:
        raise ValueError(f"Function {f} does not take any arguments.")
    f_type_hint = params[0].annotation
    args = get_args(f_type_hint)
    if not args:
        raise ValueError(f"Could not infer type from {f_type_hint}")
    iterable_arg = args[0]  # The type inside Iterable[...]
    return iterable_arg

class System:
    """
    A System wraps a function with a signature of (Ctx) -> None.
    It can be created from a function with a signature of (Iterable[T]), or (Iterable[Tuple[T1, T2]]), etc.
    Therefore, a System abstracts over different arities of functions and provides a common interface to call them.
    """

    def __init__(self, f: Callable[[Ctx], None]) -> None:
        self.__f = f

    @classmethod
    def from_callable(cls, call: Callable) -> 'System':
        parameters = inspect.signature(call).parameters
        if not parameters:
            return cls.from_nullary_func(call)
        else:
            iterable_type = infer_iterable_type_from_callable(call)
            tuple_types = get_args(iterable_type)
            if not tuple_types or len(tuple_types) == 1:
                # The function accepts an Iterable[T]
                return cls.from_unary_query_func(call)
            else:
                # The function accepts an Iterable[Tuple[...]]
                arity = len(tuple_types)
                if arity == 2:
                    return cls.from_binary_query_func(call)
                elif arity == 3:
                    return cls.from_ternary_query_func(call)
                elif arity == 4:
                    return cls.from_quaternary_query_func(call)
                else:
                    raise ValueError(f"Unsupported arity of function: {arity}")

    @classmethod
    def from_nullary_func(cls, f: Callable[[], None]) -> 'System':
        """Convert a function that takes no arguments to a System."""
        def wrapped(_ctx: Ctx) -> None:
            f()
        return cls(wrapped)

    T = TypeVar('T')
    @classmethod
    def from_unary_query_func(cls, f: Callable[[Iterable[T]], None]) -> 'System':
        """Convert a function that takes an iterable of components to a System."""
        def wrapped(ctx: Ctx) -> None:
            component_type = infer_iterable_type_from_callable(f)
            f(ctx.make_unary_query(component_type))
        return cls(wrapped)

    T1 = TypeVar('T1')
    T2 = TypeVar('T2')
    @classmethod
    def from_binary_query_func(cls, f: Callable[[Iterable[Tuple[T1, T2]]], None]) -> 'System':
        """Convert a function that takes an iterable of (Component1, Component2) to a System."""
        def wrapped(ctx: Ctx) -> None:
            iterable_arg = infer_iterable_type_from_callable(f)
            tuple_args = get_args(iterable_arg)
            if len(tuple_args) == 2:
                t1, t2 = tuple_args
                f(ctx.make_binary_query(t1, t2))
            else:
                raise ValueError(f"Unsupported function signature: {f}")
        return cls(wrapped)

    T3 = TypeVar('T3')
    @classmethod
    def from_ternary_query_func(cls, f: Callable[[Iterable[Tuple[T1, T2, T3]]], None]) -> 'System':
        """Convert a function that takes an iterable of (Component1, Component2, Component3) to a System."""
        def wrapped(ctx: Ctx) -> None:
            iterable_arg = infer_iterable_type_from_callable(f)
            tuple_args = get_args(iterable_arg)
            if len(tuple_args) == 3:
                t1, t2, t3 = tuple_args
                f(ctx.make_ternary_query(t1, t2, t3))
            else:
                raise ValueError(f"Unsupported function signature: {f}")
        return cls(wrapped)

    T4 = TypeVar('T4')
    @classmethod
    def from_quaternary_query_func(cls, f: Callable[[Iterable[Tuple[T1, T2, T3, T4]]], None]) -> 'System':
        """Convert a function that takes an iterable of (Component1, Component2, Component3, Component4) to a System."""
        def wrapped(ctx: Ctx) -> None:
            iterable_arg = infer_iterable_type_from_callable(f)
            tuple_args = get_args(iterable_arg)
            if len(tuple_args) == 4:
                t1, t2, t3, t4 = tuple_args
                f(ctx.make_quaternary_query(t1, t2, t3, t4))
            else:
                raise ValueError(f"Unsupported function signature: {f}")
        return cls(wrapped)

    def call(self, ctx: Ctx) -> None:
        self.__f(ctx)

class Scheduler:
    """
    A Scheduler runs a list of Systems in order.
    """
    def __init__(self) -> None:
        self.__systems: List[System] = []

    def add_system(self, callable: Callable) -> None:
        self.__systems.append(System.from_callable(callable))

    def run(self, ctx: Ctx) -> None:
        for system in self.__systems:
            system.call(ctx)

def examples():
    """
    Some simple examples demonstrating the usage of the ECS system.
    """
    @dataclass
    class Position:
        x: float
        y: float

    @dataclass
    class Velocity:
        x: float
        y: float

    @dataclass
    class Health:
        value: int

    @dataclass
    class Name:
        value: str

    # Create a context
    ctx = Ctx()

    # Create some entities
    e1 = ctx.add_entity()
    e2 = ctx.add_entity()
    e3 = ctx.add_entity()

    # Add components to entities
    ctx.add_component(e1, Name("Player 1"))
    ctx.add_component(e1, Position(0, 0))
    ctx.add_component(e1, Velocity(1, 1))
    ctx.add_component(e1, Health(100))

    ctx.add_component(e2, Name("Player 2"))
    ctx.add_component(e2, Position(10, 10))
    ctx.add_component(e2, Health(80))

    ctx.add_component(e3, Name("Enemy"))
    ctx.add_component(e3, Position(5, 5))
    ctx.add_component(e3, Health(50))

    # Movement system: updates positions based on velocities
    def movement_system(components: Iterable[Tuple[Position, Velocity]]):
        for position, velocity in components:
            position.x += velocity.x
            position.y += velocity.y
            print(f"Moved to ({position.x}, {position.y})")

    # Health system: prints the health of all entities with Health component
    def health_system(components: Iterable[Health]):
        for health in components:
            print(f"Entity has health {health.value}")

    # Name system: prints the name of all entities with Name component
    def name_system(components: Iterable[Name]):
        for name in components:
            print(f"Entity is named {name.value}")

    # Create a scheduler and add systems
    scheduler = Scheduler()
    scheduler.add_system(movement_system)
    scheduler.add_system(health_system)
    scheduler.add_system(name_system)

    # Run the scheduler
    print("Running ECS Example:")
    scheduler.run(ctx)

def tests() -> None:
    """
    Comprehensive tests for the ECS system.
    """
    ...

if __name__ == '__main__':
    examples()
    tests()
