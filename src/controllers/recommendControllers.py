import sys
sys.path.append('src/')
from utils.recommend import recommend



class RecommendController:
    def __init__(self, date: str):
        self.date = date

    def stock_recommend(self):
        '''
        RecommendController('2022-04-13').stock_recommend()

        return: [['3661' 1.0]['6409' 2.0]]
        '''
        return recommend(date=self.date).values


if __name__ == '__main__':
    print(RecommendController('2022-04-15').stock_recommend())
