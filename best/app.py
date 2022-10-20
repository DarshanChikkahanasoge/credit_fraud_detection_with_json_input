import pickle
from flask import Flask, request ,jsonify
from preprocessing import predict_out

app=Flask(__name__)
#model=pickle.load(open('best/model.pkl','rb'))

@app.route('/predict',methods=['POST'])
def predict():
    features=request.get_json()
    with open('best/model.bin','rb') as f_in:
        model=pickle.load(f_in)
        f_in.close()
    predictions= predict_out(features)
    pred=model.predict(predictions).tolist()
    result = {
        'Predictions':pred

    }
    return jsonify(result)


if __name__=='__main__':
    app.run(debug=True)



