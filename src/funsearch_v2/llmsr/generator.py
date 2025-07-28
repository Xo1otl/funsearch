from funsearch_v2 import gas
from funsearch_v2.llmsr.ansatz import Ansatz, Criteria


class GeminiGenerator(gas.Generator[Ansatz, Criteria]):
    async def generate(self, sample: gas.Sample[Ansatz, Criteria]) -> gas.Candidates[Ansatz]:
        # TODO: sampleのcodeたしからいい感じにプロンプト作って、あたらしいAnsatzをgeminiに考えてもらう
        codes = [ansatz.code for ansatz, _ in sample]
        raise NotImplementedError("Generator.generate is not implemented yet")
