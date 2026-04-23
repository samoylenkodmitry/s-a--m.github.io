---
layout: leetcode-entry
title: "1457. Pseudo-Palindromic Paths in a Binary Tree"
permalink: "/leetcode/problem/2024-01-24-1457-pseudo-palindromic-paths-in-a-binary-tree/"
leetcode_ui: true
entry_slug: "2024-01-24-1457-pseudo-palindromic-paths-in-a-binary-tree"
---

[1457. Pseudo-Palindromic Paths in a Binary Tree](https://leetcode.com/problems/pseudo-palindromic-paths-in-a-binary-tree/description/) medium
[blog post](https://leetcode.com/problems/pseudo-palindromic-paths-in-a-binary-tree/solutions/4617468/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24012024-1457-pseudo-palindromic?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/opD0sb6rsQ4)
![image.png](/assets/leetcode_daily_images/61c15614.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/482

#### Problem TLDR

Count can-form-a-palindrome paths root-leaf in a binary tree.

#### Intuition

Let's walk a binary tree with Depth-First Search and check the frequencies in path's numbers. To form a palindrome, only a single frequency can be odd.

#### Approach

* only odd-even matters, so we can store just boolean flags mask

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(log(n))$$

#### Code

```kotlin

  fun pseudoPalindromicPaths (root: TreeNode?): Int {
    fun dfs(n: TreeNode?, freq: Int): Int = n?.run {
      val f = freq xor (1 shl `val`)
      if (left == null && right == null) {
        if (f and (f - 1) == 0) 1 else 0
      } else dfs(left, f) + dfs(right, f)
    } ?: 0
    return dfs(root, 0)
  }

```
```rust

  pub fn pseudo_palindromic_paths (root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    fn dfs(n: &Option<Rc<RefCell<TreeNode>>>, freq: i32) -> i32 {
      n.as_ref().map_or(0, |n| {
        let n = n.borrow();
        let f = freq ^ (1 << n.val);
        dfs(&n.left, f) + dfs(&n.right, f) +
          (n.left.is_none() && n.right.is_none() && (f & (f - 1) == 0)) as i32
      })
    }
    dfs(&root, 0)
  }

```

