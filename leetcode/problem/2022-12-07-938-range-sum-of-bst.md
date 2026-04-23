---
layout: leetcode-entry
title: "938. Range Sum of BST"
permalink: "/leetcode/problem/2022-12-07-938-range-sum-of-bst/"
leetcode_ui: true
entry_slug: "2022-12-07-938-range-sum-of-bst"
---

[938. Range Sum of BST](https://leetcode.com/problems/range-sum-of-bst/description/) easy

[https://t.me/leetcode_daily_unstoppable/44](https://t.me/leetcode_daily_unstoppable/44)

```kotlin

    fun rangeSumBST(root: TreeNode?, low: Int, high: Int): Int =
	if (root == null) 0 else
		with(root) {
			(if (`val` in low..high) `val` else 0) +
				(if (`val` < low) 0 else rangeSumBST(left, low, high)) +
				(if (`val` > high) 0 else rangeSumBST(right, low, high))
		}

```

* be careful with ternary operations, better wrap them in a brackets

Space: O(log N), Time: O(R), r - is a range [low, high]

