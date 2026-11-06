"""Contracts shared by the engine, registry and execution workers."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AlgorithmSpec:
    id: str
    name: str
    title: str
    description: str
    module: str
    function: str
    execution: str = 'cpu'

    def metadata(self):
        return dict(id=self.id, name=self.name, title=self.title,
                    description=self.description, execution=self.execution)


@dataclass
class AlgorithmRun:
    items: list = field(default_factory=list)
    elapsed_ms: float = 0
    status: str = 'ok'
    error: str = ''
    worker_pid: int = 0
    executor: str = 'inline'


def extract_query(checked_products):
    for item in checked_products:
        if isinstance(item, dict):
            for key in ('query', 'checked_name', 'name', 'product_name'):
                if item.get(key):
                    return str(item[key])
    return ''
