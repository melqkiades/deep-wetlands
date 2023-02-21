import json
import time

from dotenv import load_dotenv, dotenv_values

from wetlands import generate_ndwi, generate_sar, train_model, map_wetlands


def main():
    load_dotenv()
    config = dotenv_values()
    print(json.dumps(config, indent=4))

    generate_ndwi.full_cycle()
    generate_sar.full_cycle()
    train_model.full_cycle()
    map_wetlands.full_cycle()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
