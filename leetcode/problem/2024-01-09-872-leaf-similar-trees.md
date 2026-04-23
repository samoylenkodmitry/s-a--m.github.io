---
layout: leetcode-entry
title: "872. Leaf-Similar Trees"
permalink: "/leetcode/problem/2024-01-09-872-leaf-similar-trees/"
leetcode_ui: true
entry_slug: "2024-01-09-872-leaf-similar-trees"
---

[872. Leaf-Similar Trees](https://leetcode.com/problems/leaf-similar-trees/description/) easy
[blog post](https://leetcode.com/problems/leaf-similar-trees/solutions/4532654/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/9012024-872-leaf-similar-trees?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/cjhu8RUxUuo)
![image.png](/assets/leetcode_daily_images/1332ae27.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/466

#### Problem TLDR

Are leafs sequences equal for two trees.

#### Intuition

Let's build a leafs lists and compare them.

#### Approach

Let's use recursive function.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun leafs(n: TreeNode?): List<Int> = n?.run {
    (leafs(left) + leafs(right))
    .takeIf { it.isNotEmpty() } ?: listOf(`val`)
  } ?: listOf()
  fun leafSimilar(root1: TreeNode?, root2: TreeNode?) =
    leafs(root1) == leafs(root2)

```

