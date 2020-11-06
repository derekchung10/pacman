# pacman
AI pacman application, designed to get Pacman the highest score possible.

Disclaimer: the interface and files that control the mechanics of the Pacman game are from Stanford's class CS221: Artificial Intelligence: Principles and Designs.

This program creates a Pacman AI using Pacman game states and the Expectimax algorithm to calculate the move that results in the highest expected score.

Evaluation functions are important in Game Evaluation Algorithms. We cannot possibly explore all outcomes of a game, so we use an evaluation function to predict the result based on the current state. Deep Blue and AlphaGo are some examples of AI programs that use evaluation fucntions.

I designed an evaluation function for the pacman game. To run my evaluation function, type the following line after navigating to the pacman folder:

python pacman.py -l smallClassic -p ExpectimaxAgent -a evalFn=better -q -c -n X {replace X with the number of games to play}

-q: removes graphical interface if set
-l {arg}: the map to play on
-p ExpectimaxAgent: flag used to enable the Expectimax algorithm
-a evalFn=better: flag used to enable my custom evaluation function
-n {arg}: number of games to play

My evaluation function performed better than 88% of other student's functions in terms of score. I used a linear combonation of the number of food and capsule items remaining, the current score, and the position of the nearest ghost, food, and capsule.
