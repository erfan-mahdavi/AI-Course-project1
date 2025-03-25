import random  

def generate_grid(n):  
    grid = []  
    for _ in range(n):  
        row = []  
        for _ in range(n):   
            choice = random.choice([1, -1, '!'])  
            if choice == 1:  
                num = random.randint(1, 500)   
                row.append(num)  
            elif choice == -1:  
                num = random.randint(-500, -1) 
                row.append(num)  
            else:  
                row.append('!')  
        grid.append(row)  
    return grid  

def save_grid_to_file(grid, filename):  
    with open(filename, 'w') as f:  
        for row in grid:  
            f.write(' '.join(map(str, row)) + '\n')   

if __name__ == "__main__":  
    n = int(input("Enter the size of the grid (n x n): "))  
    grid = generate_grid(n)  
    output_file = "C:\play\programming\python\\uni_courses\\AI\\AI-Course-project1\\test\\grid.txt"
    save_grid_to_file(grid, output_file)  
    print(f"Grid of size {n} x {n} has been written to '{output_file}'.")  