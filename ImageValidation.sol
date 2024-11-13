// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IImageDepository {
    function search(string calldata userInfo) external view returns (string memory, string memory);
    function authorizeUser(address user) external;
    function auth_user(address user) external view returns (bool);  // Check if a user is authorized
}

contract ImageValidation {
    address public OWNER;
    IImageDepository public repository;

    struct KeyInfo {
        string ipfsAddress;
        string scramblingParams;
    }
    
    mapping(string => KeyInfo) public results;

    event Deposit(address indexed from, uint256 value);
    event SearchPerformed(string userInfo, string ipfsAddress, string scramblingParams);

    constructor(address _repositoryAddress) {
        OWNER = msg.sender;
        repository = IImageDepository(_repositoryAddress);
    }

    // Modifier to allow only the owner or authorized users to access specific functions
    modifier onlyAuthorized() {
        require(msg.sender == OWNER || repository.auth_user(msg.sender), "Not authorized to access this function");
        _;
    }

    // Deposit function to send Ether to the contract
    function deposit() public payable returns (bool) {
        emit Deposit(msg.sender, msg.value);
        return true;
    }

    // Search function to call the repository's search and store result
    function search(string memory userInfo) public onlyAuthorized {
        (string memory ipfsAddress, string memory scramblingParams) = repository.search(userInfo);
        results[userInfo] = KeyInfo(ipfsAddress, scramblingParams);
        emit SearchPerformed(userInfo, ipfsAddress, scramblingParams);
    }

    // Get result stored in results mapping
    function getResult(string memory userInfo) public view onlyAuthorized returns (string memory, string memory) {
        KeyInfo memory result = results[userInfo];
        return (result.ipfsAddress, result.scramblingParams);
    }

    // Fallback function to receive Ether
    receive() external payable {}
}
