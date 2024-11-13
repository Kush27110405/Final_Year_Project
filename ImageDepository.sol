// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ImageDepository {
    address payable public DATA_OWNER;
    address public validationContract;
    mapping(address => bool) public auth_user;  // Mapping for authorized users
    struct KeyInfo {
        string ipfsAddress;
        string scramblingParams;
    }
    mapping(string => KeyInfo) public keys;  // Maps user information to KeyInfo struct

    event UserAuthorized(address indexed user);
    event UserRemoved(address indexed user);
    event KeyAdded(string userInfo, string ipfsAddress, string scramblingParams);
    event KeyDeleted(string userInfo);
    event ValidationContractSet(address validationContract);
    event Withdrawal(address indexed owner, uint256 amount);

    constructor() {
        DATA_OWNER = payable(msg.sender);
    }

    modifier onlyOwner() {
        require(msg.sender == DATA_OWNER, "Only DATA_OWNER can perform this action");
        _;
    }

    // Set the address of the ImageValidation contract
    function setValidationContract(address _validationContract) external onlyOwner {
        validationContract = _validationContract;
    }

    // Authorize a user
    function authorizeUser(address user) public onlyOwner returns (bool) {
        auth_user[user] = true;
        emit UserAuthorized(user);
        return true;
    }

    // Remove an authorized user
    function removeUser(address user) public onlyOwner returns (bool) {
        delete auth_user[user];
        emit UserRemoved(user);
        return true;
    }

    // Add key information
    function addKey(string memory userInfo, string memory ipfsAddress, string memory scramblingParams) public onlyOwner returns (bool) {
        keys[userInfo] = KeyInfo(ipfsAddress, scramblingParams);
        emit KeyAdded(userInfo, ipfsAddress, scramblingParams);
        return true;
    }

    // Delete key information
    function deleteKey(string memory userInfo) public onlyOwner returns (bool) {
        delete keys[userInfo];
        emit KeyDeleted(userInfo);
        return true;
    }

    // Search function to allow access by DATA_OWNER, authorized users, or validationContract
    function search(string memory userInfo) public view returns (string memory, string memory) {
        require(
            msg.sender == DATA_OWNER || auth_user[msg.sender] || msg.sender == validationContract,
            "Not authorized to access this information"
        );
        KeyInfo memory keyInfo = keys[userInfo];
        return (keyInfo.ipfsAddress, keyInfo.scramblingParams);
    }

    // Withdraw contract balance to DATA_OWNER
    function withdraw() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "No balance to withdraw");
        DATA_OWNER.transfer(balance);
        emit Withdrawal(DATA_OWNER, balance);
    }

    // Fallback function to receive Ether
    receive() external payable {}
}
