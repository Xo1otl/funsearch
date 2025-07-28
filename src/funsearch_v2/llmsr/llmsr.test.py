from funsearch_v2 import llmsr

print_jax = llmsr.Ansatz(code="def f(): print(jax)")
print_jax()

history = llmsr.JsonHistory()
# history.add(print_jax, 0.0)
item = history.sample()
print(item[0][0].code)


if __name__ == "__main__":
    print("This is a test for llmsr.gas_mock.History")
    # asyncio.run(history.commit())
