package Codivas.dao.model;

import com.amazonaws.services.dynamodbv2.datamodeling.*;
import lombok.*;

@Builder
@ToString
@DynamoDBTable(tableName = "PaymentAttributeTable")
@Data
@NoArgsConstructor
@AllArgsConstructor(access = AccessLevel.PACKAGE)
public class PaymentPojo {
    @DynamoDBAttribute(attributeName = "type")
    private String type;

    @DynamoDBHashKey(attributeName = "customerName")
    private String customerName;

    @DynamoDBRangeKey(attributeName = "destinationName")
    private String destinationName;

    @DynamoDBAttribute(attributeName = "amount")
    private float amount;

    @DynamoDBAttribute(attributeName = "oldCustomerBalance")
    private float oldCustomerBalance;

    @DynamoDBAttribute(attributeName = "newCustomerBalance")
    private float newCustomerBalance;

    @DynamoDBAttribute(attributeName = "oldDestinationBalance")
    private float oldDestinationBalance;

    @DynamoDBAttribute(attributeName = "newDestinationBalance")
    private float newDestinationBalance;
}
