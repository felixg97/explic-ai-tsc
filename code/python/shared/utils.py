"""
utility functions
"""
import os

def find_path(os_path: str, relative_path: str):
    path_els = relative_path.split('/')
    path_to_go = os_path
    for el in path_els:
        if el == '.':
            continue
        elif el == '..':
            path_to_go = os.path.dirname(path_to_go)
        else:
            path_to_go = path_to_go + f'\{ el }'
    return path_to_go
