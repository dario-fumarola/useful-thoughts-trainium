# AWS Trainium EC2: End-to-End Launch Guide

This guide covers the infrastructure workflow only: discover Trainium
instance types and regions/AZs, create/reuse SSH resources, launch an
EC2 instance, connect over SSH or VS Code, inspect instances, and
stop/terminate them. It intentionally does **not** cover Python,
PyTorch, or Neuron SDK setup.

## 0. Assumptions

-   AWS CLI is installed on your Mac.
-   You authenticate with `aws login`.
-   Commands are run from the **Mac terminal** unless stated otherwise.
-   Never share your `.pem` private key.
-   Trainium instances are billable while running.

## 1. Authenticate

Set a default region if needed:

``` bash
aws configure set region us-east-1
```

Log in:

``` bash
aws login
```

Verify:

``` bash
aws sts get-caller-identity
```

If authentication later expires, repeat `aws login`.

## 2. Discover Trainium models

Common Trainium families include:

-   `trn1.2xlarge` --- 1 Trainium accelerator
-   `trn1.32xlarge` --- 16 Trainium accelerators
-   `trn1n.32xlarge` --- 16 Trainium accelerators, enhanced networking
-   `trn2.3xlarge` --- 1 Trainium2 accelerator
-   `trn2.48xlarge` --- 16 Trainium2 accelerators
-   `trn2u.48xlarge` --- Trainium2 UltraServer instance type

Query Trainium types known to EC2:

``` bash
aws ec2 describe-instance-types \
  --include-unsupported-in-region \
  --filters "Name=instance-type,Values=trn1.*,trn1n.*,trn2.*,trn2u.*" \
  --query 'InstanceTypes[*].InstanceType' \
  --output text | tr '\t' '\n' | sort
```

You can also query a specific type:

``` bash
INSTANCE_TYPE=trn2.3xlarge

aws ec2 describe-instance-types \
  --instance-types "$INSTANCE_TYPE" \
  --include-unsupported-in-region \
  --output table
```

## 3. Find a region that offers the model

Set the model you want:

``` bash
INSTANCE_TYPE=trn2.3xlarge
```

Search all regions visible to your account:

``` bash
for region in $(aws ec2 describe-regions \
  --query 'Regions[*].RegionName' \
  --output text); do

  result=$(aws ec2 describe-instance-type-offerings \
    --region "$region" \
    --location-type region \
    --filters "Name=instance-type,Values=$INSTANCE_TYPE" \
    --query 'InstanceTypeOfferings[0].InstanceType' \
    --output text 2>/dev/null)

  if [ "$result" = "$INSTANCE_TYPE" ]; then
    echo "$INSTANCE_TYPE offered in: $region"
  fi
done
```

An offering means EC2 supports that instance type there. It does **not**
guarantee spare On-Demand capacity at launch time.

Choose one returned region:

``` bash
REGION=us-east-1
```

## 4. Find an Availability Zone

``` bash
aws ec2 describe-instance-type-offerings \
  --region "$REGION" \
  --location-type availability-zone \
  --filters "Name=instance-type,Values=$INSTANCE_TYPE" \
  --query 'InstanceTypeOfferings[*].[Location,InstanceType]' \
  --output table
```

Pick an AZ from the result:

``` bash
AZ=us-east-1f
```

## 5. Find the VPC and subnet

Find the default VPC:

``` bash
aws ec2 describe-vpcs \
  --region "$REGION" \
  --filters "Name=is-default,Values=true" \
  --query 'Vpcs[*].[VpcId,CidrBlock]' \
  --output table
```

Save its ID:

``` bash
VPC_ID=vpc-xxxxxxxx
```

Find a subnet in the required AZ:

``` bash
aws ec2 describe-subnets \
  --region "$REGION" \
  --filters \
    "Name=vpc-id,Values=$VPC_ID" \
    "Name=availability-zone,Values=$AZ" \
  --query 'Subnets[*].[SubnetId,AvailabilityZone,AvailabilityZoneId,MapPublicIpOnLaunch]' \
  --output table
```

Choose a subnet, preferably one showing `MapPublicIpOnLaunch = True` for
this simple SSH workflow:

``` bash
SUBNET_ID=subnet-xxxxxxxx
```

## 6. Create an SSH key pair

EC2 key pairs are regional. Create one in the same region as the
instance:

``` bash
KEY_NAME=ema-trainium-key

mkdir -p ~/.ssh

aws ec2 create-key-pair \
  --region "$REGION" \
  --key-name "$KEY_NAME" \
  --key-type rsa \
  --key-format pem \
  --query 'KeyMaterial' \
  --output text > ~/.ssh/${KEY_NAME}.pem

chmod 400 ~/.ssh/${KEY_NAME}.pem
```

Verify:

``` bash
aws ec2 describe-key-pairs \
  --region "$REGION" \
  --key-names "$KEY_NAME" \
  --output table
```

If you already have the correct regional key pair and its private
`.pem`, reuse it instead of creating another one.

## 7. Create an SSH security group

``` bash
SG_NAME=ema-trainium-ssh

SG_ID=$(aws ec2 create-security-group \
  --region "$REGION" \
  --group-name "$SG_NAME" \
  --description "SSH access for Trainium" \
  --vpc-id "$VPC_ID" \
  --query 'GroupId' \
  --output text)

echo "$SG_ID"
```

Allow SSH only from your current public IP:

``` bash
MY_IP=$(curl -s https://checkip.amazonaws.com)

aws ec2 authorize-security-group-ingress \
  --region "$REGION" \
  --group-id "$SG_ID" \
  --protocol tcp \
  --port 22 \
  --cidr "${MY_IP}/32"
```

If your public IP changes later, update the security-group rule.

## 8. Resolve a Neuron AMI

For the Ubuntu 24.04 Base Neuron DLAMI:

``` bash
AMI_ID=$(aws ssm get-parameter \
  --region "$REGION" \
  --name /aws/service/neuron/dlami/base/ubuntu-24.04/latest/image_id \
  --query 'Parameter.Value' \
  --output text)

echo "$AMI_ID"
```

Verify it:

``` bash
aws ec2 describe-images \
  --region "$REGION" \
  --image-ids "$AMI_ID" \
  --query 'Images[0].[ImageId,Name,Architecture,State]' \
  --output table
```

Resolving the SSM parameter each time is preferable to permanently
hard-coding an old AMI ID.

## 9. Launch

Review your variables:

``` bash
echo "Region:   $REGION"
echo "AZ:       $AZ"
echo "Type:     $INSTANCE_TYPE"
echo "AMI:      $AMI_ID"
echo "Subnet:   $SUBNET_ID"
echo "SG:       $SG_ID"
echo "Key:      $KEY_NAME"
```

Then launch **once**:

``` bash
INSTANCE_ID=$(aws ec2 run-instances \
  --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --count 1 \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --subnet-id "$SUBNET_ID" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ema-trainium}]' \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Launched: $INSTANCE_ID"
```

Every successful `run-instances` call creates another billable instance.

### Common launch errors

**`InsufficientInstanceCapacity`**

The configuration is valid, but AWS currently lacks spare On-Demand
capacity in that AZ. Try another AZ where the type is offered, another
region, retry later, or investigate Capacity Blocks/reservations.

**`Unsupported`**

The requested combination is not currently supported. Verify the
instance offering at the exact AZ/AZ ID, AMI architecture, and subnet.
Trying another supported AZ may resolve it.

**OAuth / expired authorization grant**

Re-authenticate:

``` bash
aws login
aws sts get-caller-identity
```

## 10. Wait for the instance

``` bash
aws ec2 wait instance-running \
  --region "$REGION" \
  --instance-ids "$INSTANCE_ID"
```

Then:

``` bash
aws ec2 describe-instances \
  --region "$REGION" \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].[InstanceId,InstanceType,State.Name,PublicIpAddress,Placement.AvailabilityZone]' \
  --output table
```

## 11. SSH

Get the public IP:

``` bash
TRAINIUM_IP=$(aws ec2 describe-instances \
  --region "$REGION" \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo "$TRAINIUM_IP"
```

Connect:

``` bash
ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@"$TRAINIUM_IP"
```

Once connected, a basic hardware check is:

``` bash
neuron-ls
```

## 12. VS Code Remote SSH

On the Mac, add an entry to `~/.ssh/config`:

``` text
Host trainium
    HostName YOUR_PUBLIC_IP
    User ubuntu
    IdentityFile ~/.ssh/ema-trainium-key.pem
    IdentitiesOnly yes
```

Test from Terminal:

``` bash
ssh trainium
```

Then in VS Code install **Remote - SSH**, run **Remote-SSH: Connect to
Host...**, and select `trainium`.

If the EC2 public IP changes after a stop/start, update `HostName`.

## 13. List running Trainium instances

``` bash
aws ec2 describe-instances \
  --region "$REGION" \
  --filters \
    "Name=instance-state-name,Values=pending,running" \
    "Name=instance-type,Values=trn1.*,trn1n.*,trn2.*,trn2u.*" \
  --query 'Reservations[*].Instances[*].[InstanceId,InstanceType,State.Name,PublicIpAddress,Placement.AvailabilityZone,Tags[?Key==`Name`].Value|[0]]' \
  --output table
```

## 14. Stop or terminate when finished

Stop preserves the instance/EBS volume for later use:

``` bash
aws ec2 stop-instances \
  --region "$REGION" \
  --instance-ids "$INSTANCE_ID"
```

Start it again later:

``` bash
aws ec2 start-instances \
  --region "$REGION" \
  --instance-ids "$INSTANCE_ID"
```

After a stop/start, check the public IP again because it may change.

Terminate permanently:

``` bash
aws ec2 terminate-instances \
  --region "$REGION" \
  --instance-ids "$INSTANCE_ID"
```

## Quick recipe for trying another Trainium model

For example, to try Trainium2:

``` bash
INSTANCE_TYPE=trn2.3xlarge
```

Then repeat:

1.  Search all regions for `$INSTANCE_TYPE`.
2.  Pick a region.
3.  Query the AZs offering it.
4.  Find/create networking and an SSH key in that region.
5.  Resolve the Neuron AMI in that region.
6.  Launch.
7.  If capacity is unavailable, try another supported AZ/region or a
    capacity reservation mechanism.

For a larger Trainium1:

``` bash
INSTANCE_TYPE=trn1.32xlarge
```

For a larger Trainium2:

``` bash
INSTANCE_TYPE=trn2.48xlarge
```

Do not assume that because a smaller member of a family is offered in an
AZ, the larger member is too. Query the **exact instance type** before
launch.

## Current Trainium families

At the time this guide was prepared, AWS documents these EC2 Trainium
types:

  Family                  Instance type        Accelerators
  ----------------------- ------------------ --------------
  Trainium1               `trn1.2xlarge`                  1
  Trainium1               `trn1.32xlarge`                16
  Trainium1N              `trn1n.32xlarge`               16
  Trainium2               `trn2.3xlarge`                  1
  Trainium2               `trn2.48xlarge`                16
  Trainium2 UltraServer   `trn2u.48xlarge`               16

Always query EC2 before launch because regional/AZ offerings and
capacity change over time.

## Useful AWS references

-   EC2 `describe-instance-type-offerings`:
    https://docs.aws.amazon.com/cli/latest/reference/ec2/describe-instance-type-offerings.html
-   EC2 accelerated instance specifications:
    https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html
-   EC2 `run-instances`:
    https://docs.aws.amazon.com/cli/latest/reference/ec2/run-instances.html
