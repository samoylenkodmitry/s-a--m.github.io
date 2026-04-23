---
layout: leetcode-entry
title: "938. Range Sum of BST"
permalink: "/leetcode/problem/2024-01-08-938-range-sum-of-bst/"
leetcode_ui: true
entry_slug: "2024-01-08-938-range-sum-of-bst"
---

[938. Range Sum of BST](https://leetcode.com/problems/range-sum-of-bst/description/) easy
[blog post](https://leetcode.com/problems/range-sum-of-bst/solutions/4526585/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/8012024-938-range-sum-of-bst?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/WQWp1jxNiP8)
![image.png](/assets/leetcode_daily_images/b94bb75c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/465

#### Problem TLDR

Sum of BST in range [low..high].

#### Intuition

Let's iterate it using a Depth-First Search and check if each value is in the range.

#### Approach

* Careful: if the current node is out of range, we still must visit its children.
* However, we can prune visit on the one side

#### Complexity

- Time complexity:
$$O(r)$$, r is a range

- Space complexity:
$$O(log(n))$$

#### Code

```kotlin

  fun rangeSumBST(root: TreeNode?, low: Int, high: Int): Int =
   root?.run {
      (if (`val` in low..high) `val` else 0) +
      (if (`val` > low) rangeSumBST(left, low, high) else 0) +
      (if (`val` < high) rangeSumBST(right, low, high) else 0)
    } ?: 0

```

