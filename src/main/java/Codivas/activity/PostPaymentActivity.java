package Codivas.activity;

import Codivas.component.PostPaymentComponent;
import com.amazonaws.services.lambda.runtime.Context;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import Codivas.constants.AppSyncRequest;
import Codivas.constants.PaymentAttributes;

import java.util.Map;

public class PostPaymentActivity {
    Gson gson = new GsonBuilder().setPrettyPrinting().create();
    PostPaymentComponent postPaymentComponent = new PostPaymentComponent();
    public String handleRequest(Map<String, Object> event, Context context) {
        System.out.println("Activity:");

        String jsonString = gson.toJson(event);
        AppSyncRequest payload = gson.fromJson(jsonString, AppSyncRequest.class);

        Map<String, Object> arguments = payload.getArguments();

        String inputJson = gson.toJson(arguments.get("input"));
        Map<String, Object> inputMap = gson.fromJson(inputJson, new TypeToken<Map<String, Object>>() {
        }.getType());

        String type = (String) inputMap.get("type");
        String customerName = (String) inputMap.get("customerName");
        String destinationName = (String) inputMap.get("destinationName");
        double amountDouble = (double) inputMap.get("amount");
        float amount = (float) amountDouble;
        double oldCustomerBalanceDouble = (double) inputMap.get("oldCustomerBalance");
        float oldCustomerBalance = (float) oldCustomerBalanceDouble;
        double newCustomerBalanceDouble = (double) inputMap.get("newCustomerBalance");
        float newCustomerBalance = (float) newCustomerBalanceDouble;

        PaymentAttributes paymentAttributes = new PaymentAttributes();
        paymentAttributes.setType(type);
        paymentAttributes.setAmount(amount);
        paymentAttributes.setCustomerName(customerName);
        paymentAttributes.setDestinationName(destinationName);
        paymentAttributes.setNewCustomerBalance(newCustomerBalance);
        paymentAttributes.setOldCustomerBalance(oldCustomerBalance);
        System.out.println("Request: " + gson.toJson(paymentAttributes));

        return postPaymentComponent.postPayment(paymentAttributes);
    }
}
