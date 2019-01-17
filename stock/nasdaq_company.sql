/*
 Navicat Premium Data Transfer

 Source Server         : localhost
 Source Server Type    : MySQL
 Source Server Version : 50637
 Source Host           : localhost:3306
 Source Schema         : Stock

 Target Server Type    : MySQL
 Target Server Version : 50637
 File Encoding         : 65001

 Date: 17/01/2019 19:08:35
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for nasdaq_company
-- ----------------------------
DROP TABLE IF EXISTS `nasdaq_company`;
CREATE TABLE `nasdaq_company` (
  `index` bigint(20) DEFAULT NULL,
  `Symbol` text,
  `Name` text,
  `LastSale` double DEFAULT NULL,
  `MarketCap` double DEFAULT NULL,
  `ADR TSO` double DEFAULT NULL,
  `IPOyear` double DEFAULT NULL,
  `Sector` text,
  `Industry` text,
  `Summary Quote` text,
  `Unnamed: 9` double DEFAULT NULL,
  KEY `ix_nasdaq_company_index` (`index`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

SET FOREIGN_KEY_CHECKS = 1;
