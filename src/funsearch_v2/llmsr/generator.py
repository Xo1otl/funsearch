from funsearch_v2 import genas
from funsearch_v2.llmsr.ansatz import Ansatz, Criteria


class GeminiGenerator(genas.Generator[Ansatz, Criteria]):
    async def generate(self, sample: genas.Sample[Ansatz, Criteria]) -> genas.Candidates[Ansatz]:
        # TODO: sampleのcodeたしからいい感じにプロンプト作って、あたらしいAnsatzをgeminiに考えてもらう
        codes = [ansatz.code for ansatz, _ in sample]
        raise NotImplementedError("Generator.generate is not implemented yet")
