---
layout: leetcode-entry
title: "129. Sum Root to Leaf Numbers"
permalink: "/leetcode/problem/2024-04-15-129-sum-root-to-leaf-numbers/"
leetcode_ui: true
entry_slug: "2024-04-15-129-sum-root-to-leaf-numbers"
---

[129. Sum Root to Leaf Numbers](https://leetcode.com/problems/sum-root-to-leaf-numbers/description/) medium
[blog post](https://leetcode.com/problems/sum-root-to-leaf-numbers/solutions/5025136/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15042024-129-sum-root-to-leaf-numbers?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/eEc3nRhGk5A)
![2024-04-15_07-58.webp](/assets/leetcode_daily_images/e6c0f812.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/572

#### Problem TLDR

Sum root-leaf numbers in a Binary Tree #medium

#### Intuition

Pass the number as an argument and return it on leaf nodes

#### Approach

I for now think it is impossible to reuse the method signature as-is and do it bottom up, at least you must return the power of 10 as an additional value.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(log(n))$$, for the recursion, however Morris Traversal will make it O(1)

#### Code

```kotlin

    fun sumNumbers(root: TreeNode?, n: Int = 0): Int = root?.run {
        if (left == null && right == null) n * 10 + `val` else
        sumNumbers(left, n * 10 + `val`) + sumNumbers(right, n * 10 + `val`)
    } ?: 0

```
```rust

    pub fn sum_numbers(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        fn dfs(n: &Option<Rc<RefCell<TreeNode>>>, x: i32) -> i32 {
            n.as_ref().map_or(0, |n| { let n = n.borrow();
                if n.left.is_none() && n.right.is_none() { x * 10 + n.val } else {
                    dfs(&n.left, x * 10 + n.val) + dfs(&n.right, x * 10 + n.val)
                }
            })
        }
        dfs(&root, 0)
    }

```

