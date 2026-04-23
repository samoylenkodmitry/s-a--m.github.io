---
layout: leetcode-entry
title: "543. Diameter of Binary Tree"
permalink: "/leetcode/problem/2024-02-27-543-diameter-of-binary-tree/"
leetcode_ui: true
entry_slug: "2024-02-27-543-diameter-of-binary-tree"
---

[543. Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/description/) easy
[blog post](https://leetcode.com/problems/diameter-of-binary-tree/solutions/4788208/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27022024-543-diameter-of-binary-tree?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/zRd-9S34LrY)
![2024-02-27_08-18.png](/assets/leetcode_daily_images/9b62c79e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/521

#### Problem TLDR

Max distance between any nodes in binary tree.

#### Intuition

Distance is the sum of the longest depths in left and right nodes.

#### Approach

We can return a pair of sum and max depth, but modifying an external variable looks simpler.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(log(n))$$

#### Code

```kotlin

  fun diameterOfBinaryTree(root: TreeNode?): Int {
    var max = 0
    fun dfs(n: TreeNode?): Int = n?.run {
      val l = dfs(left); val r = dfs(right)
      max = max(max, l + r); 1 + max(l, r)
    } ?: 0
    dfs(root)
    return max
  }

```
```rust

  pub fn diameter_of_binary_tree(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    let mut res = 0;
    fn dfs(n: &Option<Rc<RefCell<TreeNode>>>, res: &mut i32) -> i32 {
      n.as_ref().map_or(0, |n| { let n = n.borrow();
        let (l, r) = (dfs(&n.left, res), dfs(&n.right, res));
        *res = (*res).max(l + r); 1 + l.max(r)
      })
    }
    dfs(&root, &mut res); res
  }

```

