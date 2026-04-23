---
layout: leetcode-entry
title: "606. Construct String from Binary Tree"
permalink: "/leetcode/problem/2023-12-08-606-construct-string-from-binary-tree/"
leetcode_ui: true
entry_slug: "2023-12-08-606-construct-string-from-binary-tree"
---

[606. Construct String from Binary Tree](https://leetcode.com/problems/construct-string-from-binary-tree/description/) easy
[blog post](https://leetcode.com/problems/construct-string-from-binary-tree/solutions/4377687/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/08122023-606-construct-string-from?r=2bam17&utm_campaign=post&utm_medium=web)
[youtube](https://youtu.be/S0UF6M72Xyc)
![image.png](/assets/leetcode_daily_images/9a1225f8.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/432

#### Problem TLDR

Pre-order binary tree serialization

#### Intuition

Let's write a recursive solution.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun tree2str(root: TreeNode?): String = root?.run {
      val left = tree2str(left)
      val right = tree2str(right)
      val curr = "${`val`}"
      if (left == "" && right == "") curr
        else if (right == "") "$curr($left)"
        else "$curr($left)($right)"
    } ?: ""

```

