from dash import Dash, html, dcc, Input, Output, State
import dash_cytoscape as cyto
from visualization.kg_qa import KGQAProcessor

app = Dash(__name__)
qa_processor = KGQAProcessor("demo_graph.pkl")

app.layout = html.Div([
    cyto.Cytoscape(
        id='cytoscape',
        elements=[],
        layout={'name': 'cose'},
        style={'width': '100%', 'height': '600px'}
    ),
    dcc.Textarea(id='question-input', style={'width': '80%'}),
    html.Button('Submit', id='submit-btn')
])

@app.callback(
    Output('cytoscape', 'elements'),
    Input('submit-btn', 'n_clicks'),
    State('question-input', 'value')
)
def update_graph(n_clicks, question):
    if n_clicks and question:
        _, entities = qa_processor.process_question(question)
        elements = qa_processor.kg.get_visualization_elements()
        
        # 添加高亮样式
        for elem in elements:
            if 'id' in elem['data'] and elem['data']['id'] in entities:
                elem['styles'] = {'background-color': '#FF6B6B'}
        return elements
    return []

if __name__ == '__main__':
    app.run_server(debug=True)