def pop(rects):
    
    rects.pop(0)

def main():
    rects = range(0,10)
    for i in range(5):
        pop(rects)
        print(rects)

if __name__ == '__main__':
    main()
