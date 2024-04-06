package Codivas.dao;

import Codivas.constants.PaymentAttributes;
import Codivas.dao.model.PaymentPojo;
import com.amazonaws.services.dynamodbv2.AmazonDynamoDB;
import com.amazonaws.services.dynamodbv2.AmazonDynamoDBClientBuilder;
import com.amazonaws.services.dynamodbv2.datamodeling.DynamoDBMapper;

public class DBAccessor {
    private AmazonDynamoDB amazonDynamoDB;

    public String postPayment(PaymentAttributes paymentAttributes) {
        System.out.println("Init:");
        try {
            AmazonDynamoDB amazonDynamoDB = AmazonDynamoDBClientBuilder.standard().build();
            DynamoDBMapper dynamoDBMapper = new DynamoDBMapper(amazonDynamoDB);

            System.out.println("DB:");
            PaymentPojo paymentPojo = PaymentPojo.builder()
                    .type(paymentAttributes.getType())
                    .amount(paymentAttributes.getAmount())
                    .customerName(paymentAttributes.getCustomerName())
                    .destinationName(paymentAttributes.getDestinationName())
                    .newCustomerBalance(paymentAttributes.getNewCustomerBalance())
                    .oldCustomerBalance(paymentAttributes.getOldCustomerBalance())
                    .newDestinationBalance(paymentAttributes.getNewDestinationBalance())
                    .oldDestinationBalance(paymentAttributes.getOldDestinationBalance())
                    .build();
            dynamoDBMapper.save(paymentPojo);
            System.out.println("Saved");
            return "Successful";
        } catch (Exception exception) {
            System.out.println(exception.getMessage());
            return "Unsuccessful";
        }
    }
}
