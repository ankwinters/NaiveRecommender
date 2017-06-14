#!/usr/bin/env python


import numpy as np
from operator import itemgetter
from math import sqrt


class RecommendSystemIo:
    def __init__(self, rows=943, columns=1682):
        # Items represent data in output
        self.items = []
        # Matrix represents data in process
        self.matrix = np.zeros((rows, columns), dtype=np.int)

    def read_file(self, filename):
        with open(filename) as f:
            for i, line in enumerate(f):
                # Strip \n
                item = line[:-1].split(' ', 2)
                self.items.append(item)
        self.items_to_matrix()

    def write_file(self, outputfile):
        #print(self.items)
        self.matrix_to_items()
        with open(outputfile, 'w') as f:
            for item in self.items:
                line = item[0]+" "+item[1]+" "+item[2]+"\n"
                f.write(line)

    def fill_the_blanks(self):
        pass

    # ----------- Don't call them explicitly -------
    def items_to_matrix(self):
        """Initialize matrix"""
        for item in self.items:
            row = int(item[0])
            column = int(item[1])
            self.matrix[row - 1, column - 1] = int(item[2])

    def matrix_to_items(self):
        self.items = []
        itemlist = self.matrix.tolist()
        for row, line in enumerate(itemlist):
            for column, value in enumerate(line):
                item = [str(row+1), str(column+1), str(value)]
                self.items.append(item)

    # ------------ For debugging --------------------
    def print_cold_item(self):
        sum_by_line = np.sum(self.matrix, axis=0)
        for item_id, item_sum in enumerate(sum_by_line):
            if item_sum == 0:
                print("Cold item", item_id+1)

    def print_matrix(self):
        # Fix: python3 need binary mode(bytes)
        with open("matrix.txt", 'wb') as f:
            np.savetxt(f, self.matrix, "%d")


# ----------- Core recommend system -----------------------------
class RecommendSystem:
    def __init__(self, pref_matrix):
        self.measure = SimilarityMeasure(pref_matrix)
        self.total_users = pref_matrix.shape[0]
        self.total_items = pref_matrix.shape[1]
        # Initialize similarity matrix
        self.recommendation_matrix = pref_matrix
        self.sim_matrix = np.zeros((self.total_items, self.total_items), dtype=np.float64)
        self.similar_dict = {}

    def compute_similar_matrix(self):
        self.sim_matrix = self.measure.sim_cosine()
        with open("sim_matrix.txt", 'wb') as f:
            np.savetxt(f, self.sim_matrix, "%f")

# ------------ Get similarity matrix from file-------------------------
    def load_similarity_matrix(self, sim_file="cosine_similarity.txt"):
        with open(sim_file, 'r') as f:
            self.sim_matrix = np.loadtxt(f)

    def get_most_similar_items(self, num=3):
        for index, line in enumerate(self.sim_matrix):
            value_dict = dict((k, v) for k, v in enumerate(line))

            sorted_dict = sorted(value_dict.items(), key=itemgetter(1))
            #print(sorted_dict)
            similar_items = []
            for i in range(1, num+1):
                similar_items.append(sorted_dict[-i][0])
            self.similar_dict[index] = similar_items

    def get_recommendation_value(self, user_row, item_col):
        similar_items = self.similar_dict[item_col]
        total_score = 0
        total_sim = 0
        ranked_items = 0
        for sim_col in similar_items:
            score = self.measure.prefs[user_row, sim_col]
            if score == 0:
                continue
            similarity = self.sim_matrix[item_col, sim_col]
            total_score += score * similarity
            total_sim += similarity
            ranked_items += 1
            if ranked_items == 3:
                break
        if total_sim == 0:
            return 0
        else:
            return total_score / total_sim

    def fill_recommendation_matrix(self):
        # Enumerate original preference matrix
        for row, line in enumerate(self.measure.prefs):
            print("Fill data for user:", row)
            for column, value in enumerate(line):
                if value == 0:
                    val = self.get_recommendation_value(row, column)
                    self.recommendation_matrix[row, column] = round(val)


# ------------ Item-based similarity measurement-------------------
class SimilarityMeasure:
    def __init__(self, pref_matrix):
        self.prefs = pref_matrix
        self.user_mean = []
        self.compute_user_mean()

    # Directly compute similarity matrix
    def sim_cosine(self):
        similar_matrix = np.dot(self.prefs.T, self.prefs)
        similar_matrix = similar_matrix.astype(np.float64)
        items = similar_matrix.shape[0]
        # Compute norm for each items
        for i in range(0, items):
            similar_matrix[i, i] = sqrt(similar_matrix[i, i])
        for id1 in range(0, items):
            x = similar_matrix[id1, id1]
            for id2 in range(id1, items):
                y = similar_matrix[id2, id2]
                xy = similar_matrix[id1, id2]
                if x * y == 0:
                    cos = 0.0
                else:
                    cos = float(xy) / (x * y)
                similar_matrix[id1, id2] = cos
                similar_matrix[id2, id1] = cos
        # Set to 0
        for i in range(0, items):
            similar_matrix[i, i] = 0
        return similar_matrix

    # Use adjusted cosine similarity measurement
    def compute_user_mean(self):
        total_users = self.prefs.shape[0]
        for user_id in range(0, total_users):
            rated_num = 0
            total_rating = 0
            for rating in self.prefs[user_id, :]:
                # Invalid rating for user
                if rating == 0:
                    continue
                rated_num += 1
                total_rating += rating
            mean = total_rating/rated_num
            self.user_mean.append(mean)

    def sim_cosine_adjusted(self, item1, item2):
        pass

if __name__ == "__main__":
    recommender_io = RecommendSystemIo()
    recommender_io.read_file("./train_all_txt.txt")
    run_system = RecommendSystem(recommender_io.matrix)
    run_system.compute_similar_matrix()
    #run_system.load_similarity_matrix()
    #run_system.get_most_similar_items(100)
    #run_system.fill_recommendation_matrix()

    #recommender_io.matrix = run_system.recommendation_matrix
    #recommender_io.write_file("output_all.txt")


    #print(sim.sim_cosine(5,5))
    #print(sim.sim_cosine_adjusted(5,5))
    #recommender_io.print_cold_item()
    #recommender.print_matrix()
    #recommender.matrix_to_items()
    #recommender.write_file("./all_txt.txt")
