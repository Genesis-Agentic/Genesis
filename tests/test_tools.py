import unittest

from genesis_agentic.tools import GenesisToolFactory, ToolsFactory
from pydantic import Field, BaseModel
from llama_index.core.tools import FunctionTool


class TestToolsPackage(unittest.TestCase):
    def test_tools_factory_init(self):
        genesis_customer_id = "4584783"
        genesis_corpus_id = "4"
        genesis_api_key = "api_key"
        vec_factory = GenesisToolFactory(
            genesis_customer_id, genesis_corpus_id, genesis_api_key
        )

        self.assertEqual(genesis_customer_id, vec_factory.genesis_customer_id)
        self.assertEqual(genesis_corpus_id, vec_factory.genesis_corpus_id)
        self.assertEqual(genesis_api_key, vec_factory.genesis_api_key)

    def test_get_tools(self):
        def mult(x, y):
            return x * y

        class QueryToolArgs(BaseModel):
            query: str = Field(description="The user query")

        genesis_customer_id = "4584783"
        genesis_corpus_id = "4"
        genesis_api_key = "api_key"
        vec_factory = GenesisToolFactory(
            genesis_customer_id, genesis_corpus_id, genesis_api_key
        )

        query_tool = vec_factory.create_rag_tool(
            tool_name="rag_tool",
            tool_description="""
            Returns a response (str) to the user query based on the data in this corpus.
            """,
            tool_args_schema=QueryToolArgs,
        )

        tools_factory = ToolsFactory()
        other_tools = tools_factory.get_tools([mult])
        self.assertTrue(len(other_tools) == 1)
        self.assertIsInstance(other_tools[0], FunctionTool)
        self.assertIsInstance(query_tool, FunctionTool)
        # ... ANY OTHER TESTS WE WANT TO ENSURE THIS FUNCTIONALITY IS CORRECT


if __name__ == "__main__":
    unittest.main()
