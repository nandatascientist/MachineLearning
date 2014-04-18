function h = computeHLR(X,theta1, theta2)

a2init = sigmoid(theta1*x);
a2 = [1;a2init];
h = sigmoid(theta2*a2);

end
