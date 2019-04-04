def prime(n):
    primes = 1
    num = 2
    while primes <= n:
            mod = 1
            ptrue = True
            while mod < (num - 1):
                    if num%(num-mod) == 0:
                            ptrue = False
                            break
                    mod += 1
            if ptrue == True:
                    primes += 1
    return(num)

a=int(input("input:"))
if a%2==0:                          #prime no.
    b=int(a/2)
    print (prime(b))

elif a%2==1:                         #fibbonacci
    b=int((a-1)/2)
    d=[0]
    d[0]=1
    d[1]=1
    for k in range(2,b+1):
        d[k].append([k-1]+d[k-2])
    
    print (d[b])
        
else:
    print("invalid input")