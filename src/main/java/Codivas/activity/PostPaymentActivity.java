package Codivas.activity;

import com.amazonaws.services.lambda.runtime.Context;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import Codivas.constants.AppSyncRequest;
import Codivas.constants.PaymentAttributes;

import java.util.Map;

public class PostPaymentActivity {
    Gson gson = new GsonBuilder().setPrettyPrinting().create();

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
        int amount = (int) inputMap.get("amount");
        int oldCustomerBalance = (int) inputMap.get("oldCustomerBalance");
        int newCustomerBalance = (int) inputMap.get("newCustomerBalance");

        PaymentAttributes paymentAttributes = new PaymentAttributes();
        paymentAttributes.setType(type);
        paymentAttributes.setAmount(amount);
        paymentAttributes.setCustomerName(customerName);
        paymentAttributes.setDestinationName(destinationName);
        paymentAttributes.setNewCustomerBalance(newCustomerBalance);
        paymentAttributes.setOldCustomerBalance(oldCustomerBalance);
        System.out.println("Request: " + gson.toJson(paymentAttributes));

        return gson.toJson(paymentAttributes);
    }
}
