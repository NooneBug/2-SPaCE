from abc import ABC

class Parser(ABC):
	def cast_dataset(self, dataset: dict, config) -> dict:
		''' cast the dataset in the correct way w.r.t. the word module '''
		pass