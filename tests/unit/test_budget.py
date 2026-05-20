from contextforge.budget import TokenCounter, BudgetAllocator


def test_token_counter_approximate():
    # cl100k_base is approximate for Anthropic — test within range not exact
    counter = TokenCounter(provider="anthropic")
    tokens = counter.count("hello world")
    assert 2 <= tokens <= 4  # approximate


def test_token_counter_empty():
    counter = TokenCounter(provider="anthropic")
    assert counter.count("") == 0


def test_token_counter_long_text():
    counter = TokenCounter(provider="anthropic")
    tokens = counter.count("word " * 100)
    assert 90 <= tokens <= 120  # approximate range


def test_budget_allocator_proportional():
    allocator = BudgetAllocator()
    budgets = allocator.allocate(scores=[0.9, 0.6, 0.3], total_budget=1800)
    assert len(budgets) == 3
    assert sum(budgets) <= 1800
    assert budgets[0] > budgets[1] > budgets[2]


def test_budget_allocator_minimum():
    allocator = BudgetAllocator(min_tokens=100)
    budgets = allocator.allocate(scores=[0.9, 0.01, 0.01], total_budget=300)
    assert all(b >= 100 for b in budgets)


def test_budget_allocator_equal_scores():
    allocator = BudgetAllocator()
    budgets = allocator.allocate(scores=[0.5, 0.5], total_budget=1000)
    assert abs(budgets[0] - budgets[1]) <= 1
