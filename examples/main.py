from python_ecs import Ctx, Scheduler
import pygame
from pygame.surface import Surface
from typing import Iterable, Tuple
from dataclasses import dataclass
from enum import Enum

class Shape(Enum):
    CIRCLE = 'circle'
    SQUARE = 'square'

@dataclass
class Position:
    x: float
    y: float

@dataclass
class Velocity:
    x: float
    y: float

@dataclass
class DeltaTime:
    value: float

@dataclass
class Clock:
    value: pygame.time.Clock

def movement_system(components: Iterable[Tuple[Position, Velocity, DeltaTime]]):
    for position, velocity, dt in components:
        position.x += velocity.x * dt.value
        position.y += velocity.y * dt.value

def render_shapes_system(components: Iterable[Tuple[Position, Shape, Surface]]):
    for position, shape, surface in components:
        draw_func = getattr(pygame.draw, shape.value)
        draw_func(surface, (255, 0, 0), (position.x, position.y), 10)

def clear_screen_system(surface_query: Iterable[Surface]):
    for surface in surface_query:
        surface.fill((0, 0, 0))

def update_display_system():
    pygame.display.flip()

def tick_clock_system(query: Iterable[Tuple[Clock, DeltaTime]]):
    for clock, dt in query:
        dt.value = clock.value.tick(60) / 1000

def main():
    ctx = Ctx()
    scheduler = Scheduler()
    scheduler.add_system(clear_screen_system)
    scheduler.add_system(movement_system)
    scheduler.add_system(render_shapes_system)
    scheduler.add_system(update_display_system)
    scheduler.add_system(tick_clock_system)
    
    surface = pygame.display.set_mode((800, 600))
    clock = Clock(pygame.time.Clock())
    dt = DeltaTime(0)

    ctx.add_entity_with_components(clock, dt)
    ctx.add_entity_with_components(Position(100, 100), Velocity(10, 0), Shape.CIRCLE, surface, dt)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        scheduler.run(ctx)
        

if __name__ == "__main__":
    pygame.init()
    main()