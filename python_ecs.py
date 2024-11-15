from typing import List, Any, Callable, Iterable, Type, TypeVar, Tuple, get_args, Dict, List, Set, Union
import inspect

class EntityId(int):
    """Wrapper around int to represent an entity id."""
    def __repr__(self):
        return f'EntityId({int(self)})'

class Ctx:
    """
    Stores all entities and components in the game.
    Data is stored as a Table, i.e a list of lists.
    Row index is the entity id, and columns store components.
    Entities and Components are never removed.
    """

    def __init__(self) -> None:
        self.entities: Set[EntityId] = set()
        self.components_by_type: Dict[Type, Dict[EntityId, Any]] = {}

    def add_entity(self) -> EntityId:
        entity_id = EntityId(len(self.entities))
        self.entities.add(entity_id)
        self.add_component(entity_id, entity_id) # Add entity id as a component so it can be queried
        return entity_id
    
    def add_component(self, entity: EntityId, component: Any) -> None:
        assert entity in self.entities, f"Entity {entity} does not exist."
        component_type = type(component)
        if component_type not in self.components_by_type:
            self.components_by_type[component_type] = {}
        self.components_by_type[component_type][entity] = component

    def add_entity_with_components(self, *components: Any) -> EntityId:
        entity_id = self.add_entity()
        for component in components:
            self.add_component(entity_id, component)
        return entity_id

    T = TypeVar('T')
    def make_unary_query(self, t: Type[T]) -> Iterable[T]:
        """Returns an iterable of all components of the given type."""
        return self.components_by_type.get(t, {}).values()
    
    T1 = TypeVar('T1')
    T2 = TypeVar('T2')
    def make_binary_query(self, t1: Type[T1], t2: Type[T2]) -> Iterable[Tuple[T1, T2]]:
        """Returns an iterable of component tuples, for entities that have both types."""
        entities_with_t1 = self.components_by_type.get(t1, {})
        entities_with_t2 = self.components_by_type.get(t2, {})
        common_entities = entities_with_t1.keys() & entities_with_t2.keys()
        return ((entities_with_t1[e], entities_with_t2[e]) for e in common_entities)
    
    T3 = TypeVar('T3')
    def make_ternary_query(self, t1: Type[T1], t2: Type[T2], t3: Type[T3]) -> Iterable[Tuple[T1, T2, T3]]:
        """Returns an iterable of component tuples, for entities that have all three types."""
        entities_with_t1 = self.components_by_type.get(t1, {})
        entities_with_t2 = self.components_by_type.get(t2, {})
        entities_with_t3 = self.components_by_type.get(t3, {})
        common_entities = entities_with_t1.keys() & entities_with_t2.keys() & entities_with_t3.keys()
        return ((entities_with_t1[e], entities_with_t2[e], entities_with_t3[e]) for e in common_entities)
    
    T4 = TypeVar('T4')
    def make_quaternary_query(self, t1: Type[T1], t2: Type[T2], t3: Type[T3], t4: Type[T4]) -> Iterable[Tuple[T1, T2, T3, T4]]:
        """Returns an iterable of component tuples, for entities that have all four types."""
        entities_with_t1 = self.components_by_type.get(t1, {})
        entities_with_t2 = self.components_by_type.get(t2, {})
        entities_with_t3 = self.components_by_type.get(t3, {})
        entities_with_t4 = self.components_by_type.get(t4, {})
        common_entities = entities_with_t1.keys() & entities_with_t2.keys() & entities_with_t3.keys() & entities_with_t4.keys()
        return ((entities_with_t1[e], entities_with_t2[e], entities_with_t3[e], entities_with_t4[e]) for e in common_entities)

T = TypeVar('T')
def infer_iterable_type_from_callable(f: Callable[[Iterable[T]], None]) -> Type[T]:
    """Infer the type T from a callable that takes an Iterable[T] as an argument."""
    func_signature = inspect.signature(f)
    f_type_hint = list(func_signature.parameters.values())[0].annotation
    args = get_args(f_type_hint)
    return args[0]

class System:
    """
    A System wraps a function with a signature of (Ctx) -> None.
    It can be created from a function with a signature of (Iterable[T]) -> None, or (Iterable[Tuple[T1, T2]]) -> None, etc.
    Therefore, a System is a way of abstracting over different arities of functions and providing a common interface to call them.
    """

    def __init__(self, f: Callable[[Ctx], None]) -> None:
        self.__f = f

    @classmethod
    def from_callable(cls, call: Callable) -> 'System':
        #check if 0 args
        if not inspect.signature(call).parameters:
            return cls.from_nullary_func(call)
        #check if Iterable[T]
        i_type = infer_iterable_type_from_callable(call)
        if not get_args(i_type):
            return cls.from_unary_query_func(call)
        #check if Iterable[Tuple[T1, T2, ...]]
        arity = len(get_args(i_type))
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
        """Convert a function that takes an iterable of components to a System. The iterable yields every component of the given type."""
        def wrapped(ctx: Ctx) -> None:
            t = infer_iterable_type_from_callable(f)
            f(ctx.make_unary_query(t))
        return cls(wrapped)

    T1 = TypeVar('T1')
    T2 = TypeVar('T2')
    @classmethod
    def from_binary_query_func(cls, f: Callable[[Iterable[Tuple[T1, T2]]], None]) -> 'System':
        """Convert a function that takes an iterable of component pairs to a System. The iterable yields the components of every entity that has both types."""
        def wrapped(ctx: Ctx) -> None:
            tup = infer_iterable_type_from_callable(f)
            t1, t2 = get_args(tup)
            f(ctx.make_binary_query(t1, t2))
        return cls(wrapped)
    
    T3 = TypeVar('T3')
    @classmethod
    def from_ternary_query_func(cls, f: Callable[[Iterable[Tuple[T1, T2, T3]]], None]) -> 'System':
        """Convert a function that takes an iterable of component triplets to a System. The iterable yields the components of every entity that has all three types."""
        def wrapped(ctx: Ctx) -> None:
            tup = infer_iterable_type_from_callable(f)
            t1, t2, t3 = get_args(tup)
            f(ctx.make_ternary_query(t1, t2, t3))
        return cls(wrapped)
    
    T4 = TypeVar('T4')
    @classmethod
    def from_quaternary_query_func(cls, f: Callable[[Iterable[Tuple[T1, T2, T3, T4]]], None]) -> 'System':
        """Convert a function that takes an iterable of component quadruples to a System. The iterable yields the components of every entity that has all four types."""
        def wrapped(ctx: Ctx) -> None:
            tup = infer_iterable_type_from_callable(f)
            t1, t2, t3, t4 = get_args(tup)
            f(ctx.make_quaternary_query(t1, t2, t3, t4))
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
    from dataclasses import dataclass

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
    def movement_system(components: Iterable[Tuple[EntityId, Position, Velocity]]):
        for i, position, velocity in components:
            position.x += velocity.x
            position.y += velocity.y
            print(f"Moved {i} to ({position.x}, {position.y})")

    # Health system: prints the health of all entities with Health component
    def health_system(components: Iterable[Tuple[EntityId, Health]]):
        for i, health in components:
            print(f"{i} has health {health.value}")

    #Name system: prints the name of all entities with Name component
    def name_system(components: Iterable[Name]):
        for name in components:
            print(f"{name.value}")

    # Create a scheduler and add systems
    scheduler = Scheduler()
    scheduler.add_system(movement_system)
    scheduler.add_system(health_system)
    scheduler.add_system(name_system)

    # Run the scheduler
    print("Running ECS Example:")
    scheduler.run(ctx)

def tests():
    """
    Comprehensive tests for the ECS system.
    """
    ...

if __name__ == '__main__':
    examples()
    tests()