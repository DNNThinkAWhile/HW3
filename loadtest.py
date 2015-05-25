class Problem:
	def __init__(self, q='How _ you', a=['are', 'am', 'is']):
		self.q = q
		self.a = a
		self._ans_index = q.split().index('_')

	def get_q_ori(self):
		return self.q

	def get_q_withans(self, ans_i):
		return self.q.replace('_', self.a[ans_i])

	def __str__(self):
		return 'q: ' + self.q + '\na: ' + str(self.a)

	def ans_pos_in_sentence(self):
		return self._ans_index


class TestData:
	def __init__(self, testfile):

		self.probs=[]
		with open(testfile, 'r') as f:
			lines = f.readlines()
			ans5 = []
			for i, l in enumerate(lines):
				l = l[l.find(')') + 2:].rstrip()
				si = l.find('[')
				ei = l.find(']')
				ans = l[si + 1: ei]
				ans5.append(ans)

				# start of a problem
				if i % 5 == 0:
					q = l[:si] + '_' + l[ei + 1:]
					q = q.replace(',', '').replace('.', '')

				if i % 5 == 4:
					self.probs.append(Problem(q, ans5))
					ans5 = []

	def __getitem__(self, key):
		return self.probs[key]
	def __len__(self):
		return len(self.probs)

def test(argv):
	if (len(argv) < 2):
		print 'loadtest.py testing_data.txt'
		return

	fname = argv[1]
	tt = TestData(fname)
	print tt[0]
	print tt[1].get_q_withans(0)
	print 'problem count:', len(tt)

	print tt[1039]

if __name__ == '__main__':
	import sys
	test(sys.argv)
