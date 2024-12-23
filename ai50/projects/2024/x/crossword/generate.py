import sys
import copy

from crossword import *


class CrosswordCreator:
    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            Var: self.crossword.words.copy() for Var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont

        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size, self.crossword.height * cell_size),
            "black",
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                rect = [
                    (j * cell_size + cell_border, i * cell_size + cell_border),
                    (
                        (j + 1) * cell_size - cell_border,
                        (i + 1) * cell_size - cell_border,
                    ),
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (
                                rect[0][0] + ((interior_size - w) / 2),
                                rect[0][1] + ((interior_size - h) / 2) - 10,
                            ),
                            letters[i][j],
                            fill="black",
                            font=font,
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for Var in self.domains:
            # Iterate over copy of domains dictionary to avoid error "RuntimeError: Set changed size during iteration"
            for word in self.domains[Var].copy():
                # Remove word from domain if not same length as variable
                if len(word) != Var.length:
                    self.domains[Var].remove(word)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        # Mark as no revision made initially
        revision = False

        # Get overlap information. If no overlap, return False
        overlap = self.crossword.overlaps[x, y]
        if overlap is None:
            return False

        # Check every word in domain of x to maintain arc consistency with y
        for w1 in self.domains[x].copy():
            delete = True
            for w2 in self.domains[y]:
                if w1[overlap[0]] == w2[overlap[1]]:
                    delete = False
            if delete:
                self.domains[x].remove(w1)
                revision = True

        return revision

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        # If no arcs provided, add all arcs in problem
        if arcs is None:
            arcs = []
            for v in self.crossword.variables:
                for neighbour in self.crossword.neighbors(v):
                    arcs.append((v, neighbour))
        for arc in arcs:
            if self.revise(*arc):
                # If domain for variable is empty
                if not self.domains[arc[0]]:
                    return False
                for neighbour in self.crossword.neighbors(arc[0]):
                    arcs.append((neighbour, arc[0]))
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        for Var in self.crossword.variables:
            if Var not in assignment.keys():
                return False
        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        # Check if all values are unique
        if len(assignment) != len(set(assignment.values())):
            return False

        # Check if unary constraints met
        for key, value in assignment.items():
            if len(value) != key.length:
                return False

        # Check if binary constraints are met
        for key, value in assignment.items():
            for neighbour in self.crossword.neighbors(key):
                if neighbour in assignment.keys():
                    overlap = self.crossword.overlaps[key, neighbour]
                    if value[overlap[0]] != assignment[neighbour][overlap[1]]:
                        return False

        return True

    def order_domain_values(self, Var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """

        def eliminated(value):
            count = 0
            for neighbour in self.crossword.neighbors(Var):
                # If neighbour assigned, ignore
                if neighbour in assignment.keys():
                    break
                # If overlapping portion does not match, add to counter
                overlap = self.crossword.overlaps[Var, neighbour]
                for word in self.domains[neighbour]:
                    if value[overlap[0]] != word[overlap[1]]:
                        count += 1
            return count

        # Sort domain by ascending order of eliminated values
        return sorted(self.domains[Var], key=eliminated)

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """

        # Function to find degree of variable
        def degree(Var):
            return len(self.crossword.neighbors(Var))

        # Function to find domain size of variable
        def domainsize(Var):
            return len(self.domains[Var])

        # Get all unassigned variables
        unassigned = []
        for Var in self.crossword.variables:
            if Var not in assignment.keys():
                unassigned.append(Var)

        # Sort variables by domain size as primary key and degree as secondary key
        degree_sorted = sorted(unassigned, key=degree, reverse=True)
        domainsize_sorted = sorted(degree_sorted, key=domainsize)

        return domainsize_sorted[0]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """

        def inference(assignment):
            for Var in assignment:
                self.domains[Var] = set([assignment[Var]])
            arcs = []
            for neighbour in self.crossword.neighbors(Var):
                arcs.append((neighbour, Var))
            # If some domain becomes empty when enforcing arc consistency
            # Failure, try next value
            if not self.ac3(arcs):
                return "failed"

            inferences = {}
            for Var in self.domains:
                if len(self.domains[Var]) == 1:
                    inferences[Var] = list(self.domains[Var])[0]
            # check if any domain has only one value. add value to inferences
            return inferences

        # If assignment complete, return assignment
        if self.assignment_complete(assignment):
            return assignment

        # Pick unassigned variable
        Var = self.select_unassigned_variable(assignment)

        # Try assign value to variable from domain
        for value in self.order_domain_values(Var, assignment):
            assignment[Var] = value
            # If value not consistent, remove from assignment
            if not self.consistent(assignment):
                del assignment[Var]
                continue
            # Create copy of domains at this point
            OriginalDomains = copy.deepcopy(self.domains)

            # Enforce arc consistency
            inferences = inference(assignment)
            # If inference returns failed, try next value
            if inferences == "failed":
                del assignment[var]
                self.domains = OriginalDomains
                continue
            # Otherwise add new assignments to inferences
            assignment.update(inferences)

            # Call backtracking search on new assignment
            result = self.backtrack(assignment)
            # If backtracking search successful, return result(assignment)
            if result is not None:
                return result
            # If backtracking search failed, remove value from assignment
            # to try with another value
            del assignment[Var]
            # Restore original domain
            self.domains = original_domains
        return None


def main():
    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
