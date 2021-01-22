import argparse
import whois
import pprint 

def main():

  parser = argparse.ArgumentParser("Get whois information for a domain.")
  parser.add_argument("domain", type=str, help="The domain to query.")

  FLAGS = parser.parse_args()

  domain = whois.query(FLAGS.domain)

  pp = pprint.PrettyPrinter(indent=4)

  if domain:
    pp.pprint(domain.__dict__)
  else:
    print("Nothing found for " + FLAGS.domain)


if __name__ == '__main__':
  main()
