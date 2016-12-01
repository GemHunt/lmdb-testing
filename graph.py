'''
There is a lot to be said about using graphs to find the groups...


Grow the 1D circle mesh into a linked list:
This is how you can build the whole model without widening. This is going to suck time, but work better.
I can’t plan this too much. I need something working first.
I am going to go forward only and see how it works.
Make the out edges: You flip all the seeds so everything is going a positive angle
Make the in edges: You flip all the seeds so everything is going a negative angle (store positive)
Drop the dups (not sure about this)
You pick the starting seed with greatest total max_value and over the medians on in and out.
Pick a new seed:
One of the test images for that starting seed
That has not been used before and is not the starting seed
With an in and out score each over the median of the other nodes.
With the highest in and out score total.
?What other scoring can I use?
So I am currently ignoring what the angle is.
Remove the new seed from the out edges and put it into the circle graph
Add the angle to the circle_graph_current_angle.
Repeat 4-6 until the starting seed is found.
Don’t look for the starting seed until the circle_graph_current_angle is over 330
 Then drop the rest of the points on the circle that can be. (Plan this out when the first circle is done.)




Rate how two points are correlated through a third point:
The dup data would help here.
What is the angle between point A and B going though C
Where A->B is an edge already.
This could be A->B->C or A<-B<-C
What if it does not correlate?
A->C<-B would be an error. (I don’t care right now)


NetworkX:
Done: Model a  circle and try that.
How do I import data?
How do you move the weights to minimize the cost?
MinLA?


Gephi:
Play with it.
Install it?
Create a csv file for Gephi.
Try passing in the flipped data
Use angles for weights
How do you automatically detect outlier edges?
Can I fix only some points? Can I fix a circle?
What are the rendering options?




http://dlib.net/ml_guide.svg:  This is a structural_svm_problem


'''

import networkx as nx
import matplotlib.pyplot as plt

try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
except ImportError:
    try:
        import pydotplus
        from networkx.drawing.nx_pydot import graphviz_layout
    except ImportError:
        raise ImportError("This example needs Graphviz and either "
                          "PyGraphviz or PyDotPlus")



G = nx.star_graph(3)
L = nx.line_graph(G)
print(sorted(map(sorted, L.edges())))



G=nx.Graph()

G.add_edge('1','2',weight=30)
G.add_edge('2','3',weight=30)
G.add_edge('3','4',weight=10)
G.add_edge('4','5',weight=100)
G.add_edge('5','6',weight=60)
G.add_edge('6','7',weight=40)
G.add_edge('7','8',weight=60)
G.add_edge('8','1',weight=40)
G.add_edge('2','9',weight=15)
G.add_edge('9','3',weight=15)


elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >0.5]
esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <=0.5]

pos=nx.spring_layout(G) # positions for all nodes

# nodes
nx.draw_networkx_nodes(G,pos,node_size=700)

# edges
nx.draw_networkx_edges(G,pos,edgelist=elarge,
                    width=6)
nx.draw_networkx_edges(G,pos,edgelist=esmall,
                    width=6,alpha=0.5,edge_color='b',style='dashed')

# labels
nx.draw_networkx_labels(G,pos,font_size=20,font_family='sans-serif')

plt.axis('off')
plt.savefig("weighted_graph.png") # save as png
plt.show() # display
