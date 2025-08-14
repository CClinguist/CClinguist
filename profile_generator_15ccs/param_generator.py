import random
import json

def sample_from_dual_ranges(r1, r2):
    
    chosen_range = random.choice([r1, r2])
    return round(random.uniform(*chosen_range), 2)


def generate_params():
    # coef_obvious_penalty > coef_acc_reward > coef_search_penalty
    

    params_list = []


    for _ in range(5):
        coef_obvious_penalty_base = 20
        coef_acc_reward = 10
        coef_search_penalty = 5
        base_penalty = 15
        direction_penalty = 30
        # coef_obvious_penalty = round(random.uniform(coef_obvious_penalty_base - 5, coef_obvious_penalty_base + 5), 2)
        coef_obvious_penalty = sample_from_dual_ranges((coef_obvious_penalty_base-10,coef_obvious_penalty_base-5),
                                                       (coef_obvious_penalty_base+5,coef_obvious_penalty_base+15))

        params_list.append({
            "coef_acc_reward": coef_acc_reward,
            "coef_obvious_reward": coef_obvious_penalty,
            "coef_search_penalty": coef_search_penalty,
            "base_penalty": base_penalty,
            "direction_penalty": direction_penalty
        })

    for _ in range(5):
        coef_obvious_penalty = 20
        coef_acc_reward_base = 10
        coef_search_penalty = 5
        base_penalty = 15
        direction_penalty = 30
        # coef_acc_reward = round(random.uniform(coef_acc_reward_base - 3, coef_acc_reward_base + 3), 2)
        coef_acc_reward = sample_from_dual_ranges((coef_acc_reward_base-5,coef_acc_reward_base-1),
                                                  (coef_acc_reward_base+5,coef_acc_reward_base+10))

        params_list.append({
            "coef_acc_reward": coef_acc_reward,
            "coef_obvious_reward": coef_obvious_penalty,
            "coef_search_penalty": coef_search_penalty,
            "base_penalty": base_penalty,
            "direction_penalty": direction_penalty
        })


    for _ in range(5):
        coef_obvious_penalty = 20
        coef_acc_reward = 10
        coef_search_penalty_base = 5
        base_penalty = 15
        direction_penalty = 30
        # coef_search_penalty = round(random.uniform(coef_search_penalty_base - 2, coef_search_penalty_base + 2), 2)
        coef_search_penalty = sample_from_dual_ranges((coef_search_penalty_base-5,coef_search_penalty_base-1),
                                                      (coef_search_penalty_base+1,coef_search_penalty_base+5))
        
        params_list.append({
            "coef_acc_reward": coef_acc_reward,
            "coef_obvious_reward": coef_obvious_penalty,
            "coef_search_penalty": coef_search_penalty,
            "base_penalty": base_penalty,
            "direction_penalty": direction_penalty
        })
    

    for _ in range(5):
        coef_obvious_penalty = 20
        coef_acc_reward = 10
        coef_search_penalty = 5
        base_penalty = 15
        direction_penalty = 30

        # new_base_penalty = round(random.uniform(base_penalty - 4, base_penalty + 4), 2)
        new_base_penalty = sample_from_dual_ranges((base_penalty-5,base_penalty-1),
                                                   (base_penalty+5,base_penalty+10))
        
        params_list.append({
            "coef_acc_reward": coef_acc_reward,
            "coef_obvious_reward": coef_obvious_penalty,
            "coef_search_penalty": coef_search_penalty,
            "base_penalty": new_base_penalty,
            "direction_penalty": direction_penalty
        })

    for _ in range(5):
        coef_obvious_penalty = 20
        coef_acc_reward = 10
        coef_search_penalty = 5
        base_penalty = 15
        direction_penalty = 30

        # new_direction_penalty = round(random.uniform(direction_penalty - 8, direction_penalty + 8), 2)
        new_direction_penalty = sample_from_dual_ranges((direction_penalty-15,direction_penalty-5),
                                                        (direction_penalty+5,direction_penalty+15))
        
        params_list.append({
            "coef_acc_reward": coef_acc_reward,
            "coef_obvious_reward": coef_obvious_penalty,
            "coef_search_penalty": coef_search_penalty,
            "base_penalty": base_penalty,
            "direction_penalty": new_direction_penalty
        })
    
    
    for _ in range(5):
        coef_obvious_penalty_base = 20
        coef_acc_reward_base = 10
        coef_search_penalty_base = 5
        base_penalty = 15
        direction_penalty = 30

        coef_obvious_penalty = round(random.uniform(coef_obvious_penalty_base + 5, coef_obvious_penalty_base + 10), 2)
        coef_acc_reward = round(random.uniform(coef_acc_reward_base + 3, coef_acc_reward_base + 6), 2)
        coef_search_penalty = round(random.uniform(coef_search_penalty_base + 2, coef_search_penalty_base + 4), 2)
        

        coef_obvious_penalty_decrease = round(random.uniform(coef_obvious_penalty_base - 10, coef_obvious_penalty_base - 5), 2)
        coef_acc_reward_decrease = round(random.uniform(coef_acc_reward_base - 6, coef_acc_reward_base - 3), 2)
        coef_search_penalty_decrease = round(random.uniform(coef_search_penalty_base - 4, coef_search_penalty_base - 2), 2)
        

        if random.choice([True, False]):
            params_list.append({
                "coef_acc_reward": coef_acc_reward,
                "coef_obvious_reward": coef_obvious_penalty,
                "coef_search_penalty": coef_search_penalty,
                "base_penalty": base_penalty,
                "direction_penalty": direction_penalty
            })
        else:
            params_list.append({
                "coef_acc_reward": coef_acc_reward_decrease,
                "coef_obvious_reward": coef_obvious_penalty_decrease,
                "coef_search_penalty": coef_search_penalty_decrease,
                "base_penalty": base_penalty,
                "direction_penalty": direction_penalty
            })    
    


    return params_list


params_list = generate_params()
base_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base_dir, 'params_train.json'), 'w') as json_file:
    json.dump(params_list, json_file, indent=4)

print("Generated params_train_0605.json with the specified data.")
