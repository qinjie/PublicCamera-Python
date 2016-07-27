-- phpMyAdmin SQL Dump
-- version 4.2.7.1
-- http://www.phpmyadmin.net
--
-- Host: 127.0.0.1
-- Generation Time: Jul 14, 2015 at 09:43 PM
-- Server version: 5.6.20
-- PHP Version: 5.5.15

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `publiccamera`
--

-- --------------------------------------------------------

--
-- Table structure for table `country`
--

CREATE TABLE IF NOT EXISTS `country` (
`id` int(10) unsigned NOT NULL,
  `code` char(2) NOT NULL,
  `name` char(52) NOT NULL,
  `population` int(10) unsigned NOT NULL DEFAULT '0',
  `userId` int(10) unsigned DEFAULT NULL,
  `created` timestamp NULL DEFAULT NULL,
  `modified` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB  DEFAULT CHARSET=utf8 AUTO_INCREMENT=35 ;

--
-- Dumping data for table `country`
--

INSERT INTO `country` (`id`, `code`, `name`, `population`, `userId`, `created`, `modified`) VALUES
(4, 'CN', 'China', 1277558000, 5, '2015-05-24 08:21:23', '2015-04-27 02:11:18'),
(5, 'DE', 'Germany', 82164700, 5, '2015-05-24 08:21:23', '2015-04-27 02:11:18'),
(6, 'FR', 'France', 59225700, 5, '2015-05-24 08:21:23', '2015-04-27 02:11:18'),
(7, 'GB', 'United Kingdom', 59623400, 6, '2015-05-24 08:21:23', '2015-04-27 02:11:18'),
(8, 'IN', 'India', 1013662000, 6, '2015-05-24 08:21:23', '2015-04-27 02:11:18'),
(9, 'RU', 'Russia', 146934000, 6, '2015-05-24 08:21:23', '2015-04-27 02:11:18'),
(10, 'US', 'United States', 278357000, 6, '2015-05-24 08:21:23', '2015-04-27 02:11:18'),
(18, 'SG', 'Singapore', 12345, 6, '2015-05-24 08:21:23', '2015-04-27 02:11:18'),
(19, 'MY', 'Malaysia', 10000000, 6, '2015-05-24 08:21:23', '2015-04-27 02:11:18'),
(20, 'AB', 'Ababab', 1234, NULL, NULL, '2015-06-01 06:48:45'),
(21, 'CD', 'cdcdcd', 223344, 5, NULL, '2015-06-01 06:59:33'),
(22, 'CD', 'cdcdcd', 223344, 5, NULL, '2015-06-01 07:46:59'),
(23, 'CD', 'cdcdcd', 223344, 5, NULL, '2015-06-01 08:57:14'),
(24, 'CD', 'cdcdcd', 223344, 5, NULL, '2015-06-01 09:06:05'),
(25, 'CD', 'cdcdcd', 223344, 5, NULL, '2015-06-01 09:08:20'),
(26, 'CD', 'cdcdcd', 223344, 5, NULL, '2015-06-01 09:20:16'),
(27, 'CD', 'cdcdcd', 223344, 5, NULL, '2015-06-01 09:34:20'),
(28, 'CD', 'cdcdcd', 223344, 5, NULL, '2015-06-01 09:35:18'),
(29, 'CD', 'cdcdcd', 223344, 5, NULL, '2015-06-01 09:37:18'),
(30, 'CD', 'cdcdcd', 223344, 5, NULL, '2015-06-01 09:53:53'),
(31, 'CD', 'cdcdcd', 223344, 5, NULL, '2015-06-01 10:01:27'),
(32, 'CD', 'cd2cd2cd2', 222333, 5, '2015-06-01 10:02:40', '2015-06-01 10:02:40'),
(33, 'CD', 'cd2cd2cd2', 222333, 5, '2015-06-01 10:03:59', '2015-06-01 10:03:59'),
(34, 'CD', 'cd2cd2cd2', 222333, 5, '2015-06-01 10:05:34', '2015-06-01 10:05:34');

-- --------------------------------------------------------

--
-- Table structure for table `floor`
--

CREATE TABLE IF NOT EXISTS `floor` (
`id` int(10) unsigned NOT NULL,
  `label` varchar(50) NOT NULL DEFAULT '',
  `remark` varchar(200) DEFAULT NULL,
  `status` int(5) unsigned NOT NULL DEFAULT '1',
  `serial` varchar(32) DEFAULT '' COMMENT 'unique identifier of floor',
  `projectId` int(10) unsigned NOT NULL,
  `created` timestamp NULL DEFAULT NULL,
  `modified` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=4 ;

--
-- Dumping data for table `floor`
--

INSERT INTO `floor` (`id`, `label`, `remark`, `status`, `serial`, `projectId`, `created`, `modified`) VALUES
(1, 'Pool Side Canteen', NULL, 1, 'abcdefghijklmn', 1, NULL, '2014-12-17 14:07:54'),
(2, 'Makan Place', NULL, 1, 'qwerertyrutiyioittyuyte', 1, NULL, '2014-12-17 14:07:59'),
(3, 'Munch', NULL, 1, '', 1, NULL, '2014-12-17 14:08:02');

-- --------------------------------------------------------

--
-- Table structure for table `floordata`
--

CREATE TABLE IF NOT EXISTS `floordata` (
`id` int(10) unsigned NOT NULL,
  `floorId` int(20) unsigned NOT NULL,
  `label` varchar(10) NOT NULL,
  `marker` varchar(20) DEFAULT NULL,
  `value` varchar(50) DEFAULT NULL,
  `created` timestamp NULL DEFAULT NULL,
  `modified` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=83 ;

--
-- Dumping data for table `floordata`
--

INSERT INTO `floordata` (`id`, `floorId`, `label`, `marker`, `value`, `created`, `modified`) VALUES
(2, 1, 'CrowdPast', '07:00', '5', NULL, '2015-06-12 05:32:53'),
(3, 1, 'CrowdPast', '07:30', '10', NULL, '2015-06-12 05:32:53'),
(4, 1, 'CrowdPast', '08:00', '30', NULL, '2015-06-12 05:32:53'),
(5, 1, 'CrowdPast', '09:30', '45', NULL, '2015-06-12 05:32:53'),
(6, 1, 'CrowdPast', '08:30', '35', NULL, '2015-06-12 05:32:53'),
(7, 1, 'CrowdPast', '09:00', '40', NULL, '2015-06-12 05:32:53'),
(8, 1, 'CrowdPast', '10:00', '40', NULL, '2015-06-12 05:32:53'),
(9, 1, 'CrowdPast', '10:30', '50', NULL, '2015-06-12 05:32:53'),
(10, 1, 'CrowdPast', '11:00', '80', NULL, '2015-06-12 05:32:53'),
(11, 1, 'CrowdPast', '11:30', '85', NULL, '2015-06-12 05:32:53'),
(12, 1, 'CrowdPast', '12:00', '90', NULL, '2015-06-12 05:32:53'),
(13, 1, 'CrowdPast', '12:30', '90', NULL, '2015-06-12 05:32:53'),
(14, 1, 'CrowdPast', '13:00', '85', NULL, '2015-06-12 05:32:53'),
(15, 1, 'CrowdPast', '13:30', '80', NULL, '2015-06-12 05:32:53'),
(16, 1, 'CrowdPast', '14:00', '80', NULL, '2015-06-12 05:32:53'),
(17, 1, 'CrowdPast', '14:30', '75', NULL, '2015-06-12 05:32:53'),
(18, 1, 'CrowdPast', '15:00', '60', NULL, '2015-06-12 05:32:53'),
(19, 1, 'CrowdPast', '15:30', '40', NULL, '2015-06-12 05:32:53'),
(20, 1, 'CrowdPast', '16:00', '40', NULL, '2015-06-12 05:32:53'),
(21, 1, 'CrowdPast', '16:30', '50', NULL, '2015-06-12 05:32:53'),
(25, 1, 'CrowdPast', '17:00', '60', NULL, '2015-06-12 05:32:53'),
(26, 1, 'CrowdPast', '17:30', '65', NULL, '2015-06-12 05:32:53'),
(27, 1, 'CrowdPast', '18:00', '50', NULL, '2015-06-12 05:32:53'),
(28, 1, 'CrowdPast', '18:30', '20', NULL, '2015-06-12 05:32:53'),
(29, 1, 'CrowdPast', '19:00', '0', NULL, '2015-06-12 05:32:53'),
(30, 2, 'CrowdPast', '07:00', '10', NULL, '2015-06-12 05:32:53'),
(31, 2, 'CrowdPast', '07:30', '20', NULL, '2015-06-12 05:32:53'),
(32, 2, 'CrowdPast', '08:00', '35', NULL, '2015-06-12 05:32:53'),
(33, 2, 'CrowdPast', '09:30', '55', NULL, '2015-06-12 05:32:53'),
(34, 2, 'CrowdPast', '08:30', '30', NULL, '2015-06-12 05:32:53'),
(35, 2, 'CrowdPast', '09:00', '35', NULL, '2015-06-12 05:32:53'),
(36, 2, 'CrowdPast', '10:00', '40', NULL, '2015-06-12 05:32:53'),
(37, 2, 'CrowdPast', '10:30', '45', NULL, '2015-06-12 05:32:53'),
(38, 2, 'CrowdPast', '11:00', '70', NULL, '2015-06-12 05:32:53'),
(39, 2, 'CrowdPast', '11:30', '75', NULL, '2015-06-12 05:32:53'),
(40, 2, 'CrowdPast', '12:00', '80', NULL, '2015-06-12 05:32:53'),
(41, 2, 'CrowdPast', '12:30', '85', NULL, '2015-06-12 05:32:53'),
(42, 2, 'CrowdPast', '13:00', '70', NULL, '2015-06-12 05:32:53'),
(43, 2, 'CrowdPast', '13:30', '60', NULL, '2015-06-12 05:32:53'),
(44, 2, 'CrowdPast', '14:00', '50', NULL, '2015-06-12 05:32:53'),
(45, 2, 'CrowdPast', '14:30', '65', NULL, '2015-06-12 05:32:53'),
(46, 2, 'CrowdPast', '15:00', '65', NULL, '2015-06-12 05:32:53'),
(47, 2, 'CrowdPast', '15:30', '45', NULL, '2015-06-12 05:32:53'),
(48, 2, 'CrowdPast', '16:00', '30', NULL, '2015-06-12 05:32:53'),
(49, 2, 'CrowdPast', '16:30', '40', NULL, '2015-06-12 05:32:53'),
(50, 2, 'CrowdPast', '17:00', '50', NULL, '2015-06-12 05:32:53'),
(51, 2, 'CrowdPast', '17:30', '55', NULL, '2015-06-12 05:32:53'),
(52, 2, 'CrowdPast', '18:00', '60', NULL, '2015-06-12 05:32:53'),
(53, 2, 'CrowdPast', '18:30', '30', NULL, '2015-06-12 05:32:53'),
(54, 2, 'CrowdPast', '19:00', '10', NULL, '2015-06-12 05:32:53'),
(55, 3, 'CrowdPast', '07:00', '15', NULL, '2015-06-12 05:32:53'),
(56, 3, 'CrowdPast', '07:30', '25', NULL, '2015-06-12 05:32:53'),
(57, 3, 'CrowdPast', '08:00', '40', NULL, '2015-06-12 05:32:53'),
(58, 3, 'CrowdPast', '09:30', '60', NULL, '2015-06-12 05:32:53'),
(59, 3, 'CrowdPast', '08:30', '50', NULL, '2015-06-12 05:32:53'),
(60, 3, 'CrowdPast', '09:00', '45', NULL, '2015-06-12 05:32:53'),
(61, 3, 'CrowdPast', '10:00', '35', NULL, '2015-06-12 05:32:53'),
(62, 3, 'CrowdPast', '10:30', '45', NULL, '2015-06-12 05:32:53'),
(63, 3, 'CrowdPast', '11:00', '60', NULL, '2015-06-12 05:32:53'),
(64, 3, 'CrowdPast', '11:30', '70', NULL, '2015-06-12 05:32:53'),
(65, 3, 'CrowdPast', '12:00', '80', NULL, '2015-06-12 05:32:53'),
(66, 3, 'CrowdPast', '12:30', '90', NULL, '2015-06-12 05:32:53'),
(67, 3, 'CrowdPast', '13:00', '80', NULL, '2015-06-12 05:32:53'),
(68, 3, 'CrowdPast', '13:30', '70', NULL, '2015-06-12 05:32:53'),
(69, 3, 'CrowdPast', '14:00', '55', NULL, '2015-06-12 05:32:53'),
(70, 3, 'CrowdPast', '14:30', '60', NULL, '2015-06-12 05:32:53'),
(71, 3, 'CrowdPast', '15:00', '70', NULL, '2015-06-12 05:32:53'),
(72, 3, 'CrowdPast', '15:30', '55', NULL, '2015-06-12 05:32:53'),
(73, 3, 'CrowdPast', '16:00', '40', NULL, '2015-06-12 05:32:53'),
(74, 3, 'CrowdPast', '16:30', '45', NULL, '2015-06-12 05:32:53'),
(75, 3, 'CrowdPast', '17:00', '40', NULL, '2015-06-12 05:32:53'),
(76, 3, 'CrowdPast', '17:30', '50', NULL, '2015-06-12 05:32:53'),
(77, 3, 'CrowdPast', '18:00', '40', NULL, '2015-06-12 05:32:53'),
(78, 3, 'CrowdPast', '18:30', '20', NULL, '2015-06-12 05:32:53'),
(79, 3, 'CrowdPast', '19:00', '5', NULL, '2015-06-12 05:32:53'),
(80, 1, 'CrowdNow', '15:00', '30', NULL, '2015-06-12 05:35:40'),
(81, 1, 'CrowdNow', '15:30', '40', NULL, '2015-06-12 05:35:54'),
(82, 2, 'CrowdNow', '15:00', '35', NULL, '2015-06-12 05:36:10');

-- --------------------------------------------------------

--
-- Table structure for table `floornode`
--

CREATE TABLE IF NOT EXISTS `floornode` (
`id` int(10) unsigned NOT NULL,
  `floorId` int(10) unsigned NOT NULL,
  `nodeId` int(10) unsigned NOT NULL,
  `created` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=4 ;

--
-- Dumping data for table `floornode`
--

INSERT INTO `floornode` (`id`, `floorId`, `nodeId`, `created`) VALUES
(1, 1, 1, '2015-06-02 00:56:57'),
(2, 1, 2, '2015-06-02 00:57:14'),
(3, 2, 3, '2015-06-02 00:57:23');

-- --------------------------------------------------------

--
-- Table structure for table `floorsetting`
--

CREATE TABLE IF NOT EXISTS `floorsetting` (
`id` int(10) unsigned NOT NULL,
  `floorId` int(10) unsigned NOT NULL,
  `label` varchar(10) DEFAULT NULL,
  `value` varchar(50) DEFAULT NULL,
  `created` timestamp NULL DEFAULT NULL,
  `modified` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=19 ;

--
-- Dumping data for table `floorsetting`
--

INSERT INTO `floorsetting` (`id`, `floorId`, `label`, `value`, `created`, `modified`) VALUES
(1, 1, 'markface', '1', '2015-05-28 05:56:56', '2015-06-12 00:55:33'),
(2, 2, 'markface', '1', '2015-06-01 13:33:54', '2015-06-12 00:55:41'),
(3, 3, 'markface', '1', '2015-06-01 12:55:47', '2015-06-12 00:55:47'),
(10, 1, 'starttime', '07:00', NULL, '2015-06-12 00:57:47'),
(11, 1, 'endtime', '19:00', NULL, '2015-06-12 02:05:21'),
(12, 2, 'starttime', '07:00', NULL, '2015-06-12 00:57:56'),
(13, 2, 'endtime', '19:00', NULL, '2015-06-12 02:05:25'),
(14, 3, 'starttime', '07:00', NULL, '2015-06-12 00:58:13'),
(15, 3, 'endtime', '19:00', NULL, '2015-06-12 02:05:26'),
(16, 1, 'keephours', '24', NULL, '2015-06-12 01:27:48'),
(17, 2, 'keephours', '24', NULL, '2015-06-12 01:27:48'),
(18, 3, 'keephours', '24', NULL, '2015-06-12 01:27:54');

-- --------------------------------------------------------

--
-- Table structure for table `node`
--

CREATE TABLE IF NOT EXISTS `node` (
`id` int(10) unsigned NOT NULL,
  `label` varchar(50) NOT NULL DEFAULT '' COMMENT 'human readable name',
  `type` varchar(10) NOT NULL DEFAULT 'DEFAULT' COMMENT 'allow grouping of the node',
  `status` int(5) unsigned NOT NULL DEFAULT '1',
  `serial` varchar(32) DEFAULT '' COMMENT 'unique identifier of node',
  `projectId` int(10) unsigned NOT NULL,
  `userId` int(10) unsigned NOT NULL,
  `created` timestamp NULL DEFAULT NULL,
  `modified` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=11 ;

--
-- Dumping data for table `node`
--

INSERT INTO `node` (`id`, `label`, `type`, `status`, `serial`, `projectId`, `userId`, `created`, `modified`) VALUES
(1, 'Camera 11', 'DEFAULT', 1, '', 1, 5, NULL, '2014-12-17 14:10:49'),
(2, 'Camera 12', 'DEFAULT', 1, '', 1, 5, NULL, '2014-12-17 14:11:53'),
(3, 'Camera 13', 'DEFAULT', 1, '', 1, 5, NULL, '2014-12-17 14:11:56'),
(4, 'Camera 14', 'DEFAULT', 1, '', 1, 5, NULL, '2014-12-17 14:11:58'),
(5, 'Camera 21', 'DEFAULT', 1, '', 1, 5, NULL, '2014-12-17 14:12:49'),
(6, 'Camera 22', 'DEFAULT', 1, '', 1, 5, NULL, '2014-12-17 14:12:52'),
(7, 'Camera 23', 'DEFAULT', 1, '', 1, 5, NULL, '2014-12-17 14:12:55'),
(8, 'Camera 31', 'DEFAULT', 1, '', 1, 5, NULL, '2014-12-17 14:12:57'),
(9, 'Camera 32', 'DEFAULT', 1, '', 1, 5, NULL, '2014-12-17 14:12:59'),
(10, 'Camera 33', 'DEFAULT', 1, '', 1, 5, NULL, '2014-12-17 14:13:02');

-- --------------------------------------------------------

--
-- Table structure for table `nodedata`
--

CREATE TABLE IF NOT EXISTS `nodedata` (
`id` int(10) unsigned NOT NULL,
  `nodeId` int(10) unsigned NOT NULL,
  `label` varchar(20) NOT NULL DEFAULT '' COMMENT 'data type',
  `value` varchar(50) NOT NULL DEFAULT '' COMMENT 'data value',
  `created` timestamp NULL DEFAULT NULL,
  `modified` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=9 ;

--
-- Dumping data for table `nodedata`
--

INSERT INTO `nodedata` (`id`, `nodeId`, `label`, `value`, `created`, `modified`) VALUES
(1, 2, 'temp', '37.12', '2015-01-21 01:59:43', '2015-01-21 01:59:43'),
(2, 2, 'temp', '38.12', '2015-01-21 02:01:12', '2015-01-21 02:01:12'),
(3, 2, 'temp', '38.12', '2015-01-21 08:47:00', '2015-01-21 08:47:00'),
(4, 2, 'temp', '37.12', '2015-05-28 05:25:28', '2015-05-28 05:25:28'),
(5, 1, 'humidity', '0.8', '2015-05-28 05:30:16', '2015-05-28 05:30:16'),
(6, 1, 'humidity', '0.8', '2015-06-01 02:25:42', '2015-06-01 02:25:42'),
(7, 1, 'temp', '38.12', '2015-01-21 08:47:00', '2015-01-21 08:47:00'),
(8, 1, 'temp', '37.12', '2015-01-21 01:59:43', '2015-01-21 01:59:43');

-- --------------------------------------------------------

--
-- Table structure for table `nodefile`
--

CREATE TABLE IF NOT EXISTS `nodefile` (
`id` int(10) unsigned NOT NULL,
  `nodeId` int(10) unsigned NOT NULL,
  `label` varchar(20) NOT NULL DEFAULT '' COMMENT 'data type',
  `fileName` varchar(50) DEFAULT NULL,
  `fileType` varchar(10) DEFAULT NULL,
  `fileSize` int(10) unsigned DEFAULT NULL,
  `created` timestamp NULL DEFAULT NULL,
  `modified` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=41 ;

--
-- Dumping data for table `nodefile`
--

INSERT INTO `nodefile` (`id`, `nodeId`, `label`, `fileName`, `fileType`, `fileSize`, `created`, `modified`) VALUES
(40, 4, 'This is Test 4', '0004_20150618_105027_70202600.jpg', 'image/jpeg', 1227781, '2015-06-18 02:50:28', '2015-06-18 02:50:28');

-- --------------------------------------------------------

--
-- Table structure for table `nodesetting`
--

CREATE TABLE IF NOT EXISTS `nodesetting` (
`id` int(10) unsigned NOT NULL,
  `nodeId` int(10) unsigned NOT NULL,
  `label` varchar(10) DEFAULT NULL,
  `value` varchar(50) DEFAULT NULL,
  `created` timestamp NULL DEFAULT NULL,
  `modified` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=8 ;

--
-- Dumping data for table `nodesetting`
--

INSERT INTO `nodesetting` (`id`, `nodeId`, `label`, `value`, `created`, `modified`) VALUES
(1, 1, 'Interval', '5', '2015-05-28 05:56:56', '2015-05-28 05:56:56'),
(2, 1, 'IP', '::1', '2015-06-01 13:33:54', '2015-06-05 08:41:53'),
(3, 1, 'temp', '33.33', '2015-06-01 12:55:47', '2015-06-01 12:55:47'),
(4, 1, 'sleep', '1', '2015-06-01 12:57:22', '2015-06-01 12:57:22'),
(5, 1, 'sleep', '1', '2015-06-01 13:08:47', '2015-06-01 13:08:47'),
(6, 1, 'sleep', '1', '2015-06-01 13:09:46', '2015-06-01 13:09:46'),
(7, 1, 'sleep', '1', '2015-06-01 13:32:36', '2015-06-01 13:32:36');

-- --------------------------------------------------------

--
-- Table structure for table `person`
--

CREATE TABLE IF NOT EXISTS `person` (
`id` int(11) NOT NULL COMMENT 'Unique person identifier',
  `firstName` varchar(60) NOT NULL COMMENT 'First name',
  `lastName` varchar(60) NOT NULL COMMENT 'Last name',
  `parentId` int(11) unsigned DEFAULT NULL,
  `countryId` int(11) unsigned DEFAULT NULL COMMENT 'Residing Country',
  `created` timestamp NULL DEFAULT NULL,
  `modified` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='Person master table' AUTO_INCREMENT=1 ;

-- --------------------------------------------------------

--
-- Table structure for table `project`
--

CREATE TABLE IF NOT EXISTS `project` (
`id` int(10) unsigned NOT NULL,
  `label` varchar(100) NOT NULL DEFAULT '',
  `remark` varchar(200) DEFAULT NULL,
  `serial` char(32) DEFAULT NULL COMMENT 'autogenerated, uniquely identify a project',
  `status` int(5) unsigned DEFAULT NULL,
  `userId` int(10) unsigned DEFAULT NULL,
  `created` timestamp NULL DEFAULT NULL,
  `modified` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=2 ;

--
-- Dumping data for table `project`
--

INSERT INTO `project` (`id`, `label`, `remark`, `serial`, `status`, `userId`, `created`, `modified`) VALUES
(1, 'Canteen Crowd Monitoring System', NULL, NULL, 1, 1, '2014-07-23 15:22:50', '2014-12-17 13:54:44');

-- --------------------------------------------------------

--
-- Table structure for table `projectsetting`
--

CREATE TABLE IF NOT EXISTS `projectsetting` (
`id` int(10) unsigned NOT NULL,
  `projectId` int(10) unsigned NOT NULL,
  `label` varchar(10) DEFAULT NULL,
  `value` varchar(50) DEFAULT NULL,
  `created` timestamp NULL DEFAULT NULL,
  `modified` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=33 ;

--
-- Dumping data for table `projectsetting`
--

INSERT INTO `projectsetting` (`id`, `projectId`, `label`, `value`, `created`, `modified`) VALUES
(8, 1, 'Timing', '07:00', NULL, '2015-06-12 02:55:15'),
(9, 1, 'Timing', '07:30', NULL, '2015-06-12 02:55:15'),
(10, 1, 'Timing', '08:00', NULL, '2015-06-12 02:55:15'),
(11, 1, 'Timing', '09:30', NULL, '2015-06-12 02:55:15'),
(12, 1, 'Timing', '08:30', NULL, '2015-06-12 02:55:15'),
(13, 1, 'Timing', '09:00', NULL, '2015-06-12 02:55:15'),
(14, 1, 'Timing', '10:00', NULL, '2015-06-12 02:55:15'),
(15, 1, 'Timing', '10:30', NULL, '2015-06-12 02:55:15'),
(16, 1, 'Timing', '11:00', NULL, '2015-06-12 02:55:15'),
(17, 1, 'Timing', '11:30', NULL, '2015-06-12 02:55:15'),
(18, 1, 'Timing', '12:00', NULL, '2015-06-12 02:55:15'),
(19, 1, 'Timing', '12:30', NULL, '2015-06-12 02:55:15'),
(20, 1, 'Timing', '13:00', NULL, '2015-06-12 02:55:15'),
(21, 1, 'Timing', '13:30', NULL, '2015-06-12 02:55:15'),
(22, 1, 'Timing', '14:00', NULL, '2015-06-12 02:55:15'),
(23, 1, 'Timing', '14:30', NULL, '2015-06-12 02:55:15'),
(24, 1, 'Timing', '15:00', NULL, '2015-06-12 02:55:15'),
(25, 1, 'Timing', '15:30', NULL, '2015-06-12 02:55:15'),
(26, 1, 'Timing', '16:00', NULL, '2015-06-12 02:55:15'),
(27, 1, 'Timing', '16:30', NULL, '2015-06-12 02:55:15'),
(28, 1, 'Timing', '17:00', NULL, '2015-06-12 02:55:15'),
(29, 1, 'Timing', '17:30', NULL, '2015-06-12 02:55:15'),
(30, 1, 'Timing', '18:00', NULL, '2015-06-12 02:55:15'),
(31, 1, 'Timing', '18:30', NULL, '2015-06-12 02:55:15'),
(32, 1, 'Timing', '19:00', NULL, '2015-06-12 02:55:15');

-- --------------------------------------------------------

--
-- Table structure for table `projectuser`
--

CREATE TABLE IF NOT EXISTS `projectuser` (
`id` int(10) unsigned NOT NULL,
  `projectId` int(10) unsigned NOT NULL,
  `userId` int(10) unsigned NOT NULL,
  `created` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=3 ;

--
-- Dumping data for table `projectuser`
--

INSERT INTO `projectuser` (`id`, `projectId`, `userId`, `created`) VALUES
(2, 1, 4, '2015-01-04 01:36:48');

-- --------------------------------------------------------

--
-- Table structure for table `user`
--

CREATE TABLE IF NOT EXISTS `user` (
`id` int(10) unsigned NOT NULL,
  `username` varchar(255) NOT NULL,
  `auth_key` varchar(32) DEFAULT '',
  `password_hash` varchar(255) DEFAULT '',
  `access_token` varchar(32) DEFAULT NULL,
  `password_reset_token` varchar(255) DEFAULT NULL,
  `email` varchar(255) DEFAULT '',
  `email_confirm_token` varchar(255) DEFAULT NULL,
  `role` int(10) unsigned DEFAULT '10',
  `status` smallint(6) DEFAULT '10',
  `allowance` int(10) unsigned DEFAULT NULL,
  `timestamp` int(10) unsigned DEFAULT NULL,
  `created_at` int(10) unsigned DEFAULT NULL,
  `updated_at` int(10) unsigned DEFAULT NULL
) ENGINE=InnoDB  DEFAULT CHARSET=utf8 AUTO_INCREMENT=7 ;

--
-- Dumping data for table `user`
--

INSERT INTO `user` (`id`, `username`, `auth_key`, `password_hash`, `access_token`, `password_reset_token`, `email`, `email_confirm_token`, `role`, `status`, `allowance`, `timestamp`, `created_at`, `updated_at`) VALUES
(1, 'master', 'auth-key-test-admin', '$2y$10$vsK92gjucpYK7MP.6w9Pk.N01/uH.EPaHHwnVYEAcSCjNruZ/YTPK', 'abcd1234', NULL, 'zqi2@np.edu.sg', NULL, 40, 10, NULL, NULL, 0, 0),
(2, 'admin', 'auth-key-test-admin', '$2y$10$vsK92gjucpYK7MP.6w9Pk.N01/uH.EPaHHwnVYEAcSCjNruZ/YTPK', 'abcd1234', NULL, 'zqi2@np.edu.sg', NULL, 30, 10, NULL, NULL, 0, 0),
(3, 'manager1', 'auth-key-test-admin', '$2y$10$vsK92gjucpYK7MP.6w9Pk.N01/uH.EPaHHwnVYEAcSCjNruZ/YTPK', 'abcd1234', NULL, 'zqi2@np.edu.sg', NULL, 20, 10, 299, 1434595828, 0, 0),
(4, 'manager2', 'auth-key-test-admin', '$2y$10$vsK92gjucpYK7MP.6w9Pk.N01/uH.EPaHHwnVYEAcSCjNruZ/YTPK', 'abcd1234', NULL, 'zqi2@np.edu.sg', NULL, 20, 10, 299, 1432481401, 0, 0),
(5, 'user1', 'auth-key-test-admin', '$2y$10$vsK92gjucpYK7MP.6w9Pk.N01/uH.EPaHHwnVYEAcSCjNruZ/YTPK', 'abcd1234', NULL, 'zqi2@np.edu.sg', NULL, 10, 10, 297, 1434595827, 0, 0),
(6, 'user2', 'auth-key-test-admin', '$2y$10$vsK92gjucpYK7MP.6w9Pk.N01/uH.EPaHHwnVYEAcSCjNruZ/YTPK', 'abcd1234', NULL, 'zqi2@np.edu.sg', NULL, 10, 10, 299, 1432560400, 0, 0);

-- --------------------------------------------------------

--
-- Table structure for table `usertoken`
--

CREATE TABLE IF NOT EXISTS `usertoken` (
`id` int(10) unsigned NOT NULL,
  `userId` int(10) unsigned NOT NULL,
  `token` varchar(32) NOT NULL DEFAULT '',
  `label` varchar(10) DEFAULT NULL,
  `ipAddress` varchar(32) DEFAULT NULL,
  `expire` timestamp NULL DEFAULT '0000-00-00 00:00:00',
  `created` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=48 ;

--
-- Dumping data for table `usertoken`
--

INSERT INTO `usertoken` (`id`, `userId`, `token`, `label`, `ipAddress`, `expire`, `created`) VALUES
(35, 4, 'b1eca42adda9abd921194cdc83424f8f', 'ACCESS', '::1', '2015-02-19 15:11:59', '2015-01-20 15:11:59'),
(44, 6, 'e67f2a8eca0bf67e9453014da1c1b210', 'VERIFY', '::1', '2015-06-09 03:48:18', '2015-05-10 09:48:18'),
(45, 19, '9cc9c8a73bce6979a2cc21be6581bf5a', 'VERIFY', '::1', '2015-02-19 16:06:47', '2015-01-20 16:06:47'),
(47, 1, '5f835c19a8a634b49c459aba571759c9', 'ACCESS', '::1', '2015-02-20 08:52:28', '2015-01-21 08:52:28');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `country`
--
ALTER TABLE `country`
 ADD PRIMARY KEY (`id`), ADD KEY `userId` (`userId`);

--
-- Indexes for table `floor`
--
ALTER TABLE `floor`
 ADD PRIMARY KEY (`id`), ADD KEY `projectId` (`projectId`);

--
-- Indexes for table `floordata`
--
ALTER TABLE `floordata`
 ADD PRIMARY KEY (`id`), ADD KEY `floorId` (`floorId`);

--
-- Indexes for table `floornode`
--
ALTER TABLE `floornode`
 ADD PRIMARY KEY (`id`), ADD KEY `floorId` (`floorId`), ADD KEY `nodeId` (`nodeId`);

--
-- Indexes for table `floorsetting`
--
ALTER TABLE `floorsetting`
 ADD PRIMARY KEY (`id`), ADD KEY `nodeId` (`floorId`);

--
-- Indexes for table `node`
--
ALTER TABLE `node`
 ADD PRIMARY KEY (`id`), ADD KEY `projectId` (`projectId`);

--
-- Indexes for table `nodedata`
--
ALTER TABLE `nodedata`
 ADD PRIMARY KEY (`id`), ADD KEY `nodeId` (`nodeId`);

--
-- Indexes for table `nodefile`
--
ALTER TABLE `nodefile`
 ADD PRIMARY KEY (`id`), ADD KEY `nodeId` (`nodeId`);

--
-- Indexes for table `nodesetting`
--
ALTER TABLE `nodesetting`
 ADD PRIMARY KEY (`id`), ADD KEY `nodeId` (`nodeId`);

--
-- Indexes for table `person`
--
ALTER TABLE `person`
 ADD PRIMARY KEY (`id`), ADD KEY `countryId` (`countryId`);

--
-- Indexes for table `project`
--
ALTER TABLE `project`
 ADD PRIMARY KEY (`id`), ADD KEY `ownerId` (`userId`);

--
-- Indexes for table `projectsetting`
--
ALTER TABLE `projectsetting`
 ADD PRIMARY KEY (`id`), ADD KEY `projectId` (`projectId`);

--
-- Indexes for table `projectuser`
--
ALTER TABLE `projectuser`
 ADD PRIMARY KEY (`id`), ADD KEY `projectId` (`projectId`), ADD KEY `userId` (`userId`);

--
-- Indexes for table `user`
--
ALTER TABLE `user`
 ADD PRIMARY KEY (`id`);

--
-- Indexes for table `usertoken`
--
ALTER TABLE `usertoken`
 ADD PRIMARY KEY (`id`), ADD UNIQUE KEY `token` (`token`), ADD KEY `userId` (`userId`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `country`
--
ALTER TABLE `country`
MODIFY `id` int(10) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=35;
--
-- AUTO_INCREMENT for table `floor`
--
ALTER TABLE `floor`
MODIFY `id` int(10) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=4;
--
-- AUTO_INCREMENT for table `floordata`
--
ALTER TABLE `floordata`
MODIFY `id` int(10) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=83;
--
-- AUTO_INCREMENT for table `floornode`
--
ALTER TABLE `floornode`
MODIFY `id` int(10) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=4;
--
-- AUTO_INCREMENT for table `floorsetting`
--
ALTER TABLE `floorsetting`
MODIFY `id` int(10) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=19;
--
-- AUTO_INCREMENT for table `node`
--
ALTER TABLE `node`
MODIFY `id` int(10) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=11;
--
-- AUTO_INCREMENT for table `nodedata`
--
ALTER TABLE `nodedata`
MODIFY `id` int(10) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=9;
--
-- AUTO_INCREMENT for table `nodefile`
--
ALTER TABLE `nodefile`
MODIFY `id` int(10) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=41;
--
-- AUTO_INCREMENT for table `nodesetting`
--
ALTER TABLE `nodesetting`
MODIFY `id` int(10) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=8;
--
-- AUTO_INCREMENT for table `person`
--
ALTER TABLE `person`
MODIFY `id` int(11) NOT NULL AUTO_INCREMENT COMMENT 'Unique person identifier';
--
-- AUTO_INCREMENT for table `project`
--
ALTER TABLE `project`
MODIFY `id` int(10) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=2;
--
-- AUTO_INCREMENT for table `projectsetting`
--
ALTER TABLE `projectsetting`
MODIFY `id` int(10) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=33;
--
-- AUTO_INCREMENT for table `projectuser`
--
ALTER TABLE `projectuser`
MODIFY `id` int(10) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=3;
--
-- AUTO_INCREMENT for table `user`
--
ALTER TABLE `user`
MODIFY `id` int(10) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=7;
--
-- AUTO_INCREMENT for table `usertoken`
--
ALTER TABLE `usertoken`
MODIFY `id` int(10) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=48;
--
-- Constraints for dumped tables
--

--
-- Constraints for table `country`
--
ALTER TABLE `country`
ADD CONSTRAINT `country_ibfk_1` FOREIGN KEY (`userId`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `floor`
--
ALTER TABLE `floor`
ADD CONSTRAINT `floor_ibfk_1` FOREIGN KEY (`projectId`) REFERENCES `project` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `floordata`
--
ALTER TABLE `floordata`
ADD CONSTRAINT `floordata_ibfk_1` FOREIGN KEY (`floorId`) REFERENCES `floor` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `floornode`
--
ALTER TABLE `floornode`
ADD CONSTRAINT `floornode_ibfk_1` FOREIGN KEY (`floorId`) REFERENCES `floor` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
ADD CONSTRAINT `floornode_ibfk_2` FOREIGN KEY (`nodeId`) REFERENCES `node` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `floorsetting`
--
ALTER TABLE `floorsetting`
ADD CONSTRAINT `floorsetting_ibfk_1` FOREIGN KEY (`floorId`) REFERENCES `floor` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `node`
--
ALTER TABLE `node`
ADD CONSTRAINT `node_ibfk_1` FOREIGN KEY (`projectId`) REFERENCES `project` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `nodedata`
--
ALTER TABLE `nodedata`
ADD CONSTRAINT `nodedata_ibfk_2` FOREIGN KEY (`nodeId`) REFERENCES `node` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `nodefile`
--
ALTER TABLE `nodefile`
ADD CONSTRAINT `nodefile_ibfk_2` FOREIGN KEY (`nodeId`) REFERENCES `node` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `nodesetting`
--
ALTER TABLE `nodesetting`
ADD CONSTRAINT `nodesetting_ibfk_1` FOREIGN KEY (`nodeId`) REFERENCES `node` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `person`
--
ALTER TABLE `person`
ADD CONSTRAINT `person_ibfk_1` FOREIGN KEY (`countryId`) REFERENCES `country` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `project`
--
ALTER TABLE `project`
ADD CONSTRAINT `project_ibfk_1` FOREIGN KEY (`userId`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `projectsetting`
--
ALTER TABLE `projectsetting`
ADD CONSTRAINT `projectsetting_ibfk_1` FOREIGN KEY (`projectId`) REFERENCES `project` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `projectuser`
--
ALTER TABLE `projectuser`
ADD CONSTRAINT `projectuser_ibfk_1` FOREIGN KEY (`projectId`) REFERENCES `project` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
ADD CONSTRAINT `projectuser_ibfk_2` FOREIGN KEY (`userId`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `usertoken`
--
ALTER TABLE `usertoken`
ADD CONSTRAINT `usertoken_ibfk_1` FOREIGN KEY (`userId`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
