---
layout: leetcode-entry
title: "501. Find Mode in Binary Search Tree"
permalink: "/leetcode/problem/2023-11-01-501-find-mode-in-binary-search-tree/"
leetcode_ui: true
entry_slug: "2023-11-01-501-find-mode-in-binary-search-tree"
---

[501. Find Mode in Binary Search Tree](https://leetcode.com/problems/find-mode-in-binary-search-tree/description/) easy
[blog post](https://leetcode.com/problems/find-mode-in-binary-search-tree/solutions/4233545/kotlin-in-order-traversal/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/01112023-501-find-mode-in-binary?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/5e09a4b1.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/389

#### Problem TLDR

Most frequent elements in a Binary Search Tree

#### Intuition

A simple solution is to use a `frequency` map.
Another way is the linear scan of the increasing sequence. For example, `1 1 1 2 2 2 3 3 4 4 4`: we can use one counter and drop the previous result if counter is more than the previous max.

#### Approach

To convert the Binary Search Tree into an increasing sequence, we can do an in-order traversal.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$, result can be `n` if numbers are unique

#### Code

```kotlin
    fun findMode(root: TreeNode?): IntArray {
      val res = mutableListOf<Int>()
      var maxCount = 0
      var count = 0
      var prev = Int.MAX_VALUE
      fun dfs(n: TreeNode) {
        n.left?.let { dfs(it) }
        if (prev == n.`val`) {
          count++
        } else {
          count = 1
          prev = n.`val`
        }
        if (count == maxCount) {
          res += n.`val`
        } else if (count > maxCount) {
          maxCount = count
          res.clear()
          res += n.`val`
        }
        n.right?.let { dfs(it) }
      }
      root?.let { dfs(it) }
      return res.toIntArray()
    }

```

