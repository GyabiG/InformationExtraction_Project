"""Creates a hidden markov model that can learn on Gold labled data"""
class HMM():
    def __init__(self, states):
        self.states = states
        #states could be Definition, Term, Other
        self.states.append("start")
        self.states.append("end")

    def learn(self, gold_data):
        """gold_data in form 
            word label"""
        print("ahhh")