import os
import random
import re
import sys


DAMPING = 0.85
SAMPLES = 10000


def main():
    """Main function to run pagerank algorithm"""
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from files (HTML)
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Include only links to other pages within the corpus
    for filename in pages:
        pages[filename] = set(link for link in pages[filename] if link in pages)

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.

    If a page has no outgoing links, returns an equal probability for all pages in the corpus
    """

    # Prepare the directory for the probability distribution:
    probdist = {pagename: 0 for pagename in corpus}

    # If page has no links, return equal probability for the corpus:
    if len(corpus[page]) == 0:
        for pagename in probdist:
            probdist[pagename] = 1 / len(corpus)
        return probdist

    # Likelihood of randomly selecting any page:
    randomprob = (1 - damping_factor) / len(corpus)

    # Likelihood of selecting a link from the page:
    link_prob = damping_factor / len(corpus[page])

    # Include probabilities in the distribution:
    for pagename in probdist:
        probdist[pagename] += randomprob

        if pagename in corpus[page]:
            probdist[pagename] += link_prob

    return probdist


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    visits = {pagename: 0 for pagename in corpus}

    # The first page is chosen randomly:
    current_page = random.choice(list(visits))
    visits[current_page] += 1

    # For the remaining n-1 samples, select the page according to the transistion model:
    for i in range(0, n - 1):
        trans_model = transition_model(corpus, current_page, damping_factor)

        # Select the next page according to the probabilities defined by the transition model :
        rand_val = random.random()
        total_prob = 0

        for pagename, probability in trans_model.items():
            total_prob += probability
            if rand_val <= total_prob:
                current_page = pagename
                break

        visits[current_page] += 1

    # Normalize the visits based on the sample number:
    pageranks = {pagename: (visit_num / n) for pagename, visit_num in visits.items()}

    print("Sum of sample page ranks: ", round(sum(pageranks.values()), 4))

    return pageranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Compute certain constants from the corpus for later use:
    numpages = len(corpus)
    init_rank = 1 / numpages
    random_choice_prob = (1 - damping_factor) / len(corpus)
    iterations = 0

    # The initial pagerank assigns each page a rank of 1 divided by the total number of pages in the corpus
    pageranks = {pagename: init_rank for pagename in corpus}
    newranks = {pagename: None for pagename in corpus}
    max_rank_change = init_rank

    # Repeatedly compute the pagerank until all changes are <= 0.001
    while max_rank_change > 0.001:
        iterations += 1
        max_rank_change = 0

        for pagename in corpus:
            surf_choice_prob = 0
            for other_page in corpus:
                # If another page lacks links, it randomly selects a page from the corpus:
                if len(corpus[other_page]) == 0:
                    surf_choice_prob += pageranks[other_page] * init_rank
                # Otherwise, if other_page links to page_name, it randomly selects from all the links on other_page:
                elif pagename in corpus[other_page]:
                    surf_choice_prob += pageranks[other_page] / len(corpus[other_page])
            # Compute the updated page rank
            newrank = random_choice_prob + (damping_factor * surf_choice_prob)
            newranks[pagename] = newrank

        # Normalise the updated page ranks:
        normfactor = sum(newranks.values())
        newranks = {page: (rank / normfactor) for page, rank in newranks.items()}

        # Determine the max varoation in page rank:
        for pagename in corpus:
            rank_change = abs(pageranks[pagename] - newranks[pagename])
            if rank_change > max_rank_change:
                max_rank_change = rank_change

        # Adjust page ranks to reflect the updated values:
        pageranks = newranks.copy()

    print("Iteration took", iterations, "iterations to converge")
    print("Sum of iteration page ranks: ", round(sum(pageranks.values()), 4))

    return pageranks


if __name__ == "__main__":
    main()
