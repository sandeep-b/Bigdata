def gradcheck_naive(f, x):
    """
    Gradient check for a function f
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """

    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        ### try modifying x[ix] with h defined above to compute numerical gradients
        ### make sure you call random.setstate(rndstate) before calling f(x) each time, this will make it
        ### possible to test cost functions with built in randomness later
        ### YOUR CODE HERE:

        random.setstate(rndstate)
        tmp1 = np.copy(x)
        tmp1[ix] = tmp1[ix] + h
        f1, _ = f(tmp1)

        random.setstate(rndstate)
        tmp2 = np.copy(x)
        tmp2[ix] = tmp2[ix] - h
        f2, _ = f(tmp2)
        numgrad = (f1 - f2) / (2 * h)

        ### END YOUR CODE

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print("Gradient check failed.")
            print("First gradient error found at index %s" % str(ix))
            print("Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad))
            return

        it.iternext() # Step to next dimension

    print("Gradient check passed!")
