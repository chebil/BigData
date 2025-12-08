# Jupyter Notebook Tips & Tricks

## Keyboard Shortcuts

### Command Mode (press Esc)
- `A`: Insert cell above
- `B`: Insert cell below
- `D, D`: Delete cell
- `M`: Change to Markdown
- `Y`: Change to Code
- `Shift + Enter`: Run cell

### Edit Mode (press Enter)
- `Tab`: Code completion
- `Shift + Tab`: Function documentation
- `Ctrl + ]`: Indent
- `Ctrl + [`: Dedent

## Magic Commands

```python
# Timing
%time statement  # Single execution
%%time  # Entire cell
%timeit statement  # Multiple runs

# System commands
!pip install package
!ls

# Load external scripts
%load script.py

# Display plots inline
%matplotlib inline
```

## Best Practices

1. One concept per cell
2. Use Markdown for explanations
3. Restart kernel regularly
4. Clear output before committing
5. Use cell numbers to track execution order
