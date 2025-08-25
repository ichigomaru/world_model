from rich import print
from rich.tree import Tree

def print_model_structure(model):
    tree = Tree(f"[bold cyan]Model: {model.__class__.__name__}[/bold cyan]")
    
    def add_layers(tree, module, prefix=""):
        for name, layer in module.named_children():
            subtree = tree.add(f"[bold green]{prefix}{name}[/bold green]: {layer.__class__.__name__}")
            add_layers(subtree, layer, prefix=prefix + name + ".")

    add_layers(tree, model)
    print(tree)