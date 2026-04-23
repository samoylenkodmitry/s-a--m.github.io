---
layout: leetcode-entry
title: "404. Sum of Left Leaves"
permalink: "/leetcode/problem/2024-04-14-404-sum-of-left-leaves/"
leetcode_ui: true
entry_slug: "2024-04-14-404-sum-of-left-leaves"
---

[404. Sum of Left Leaves](https://leetcode.com/problems/sum-of-left-leaves/description/) easy
[blog post](https://leetcode.com/problems/sum-of-left-leaves/solutions/5020111/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14042024-404-sum-of-left-leaves?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/TMh6FK8QmZc)
![2024-04-14_08-17.webp](/assets/leetcode_daily_images/13858a7a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/571

#### Problem TLDR

Left-leaf sum in a Binary Tree #easy

#### Intuition

Do a Depth-First Search and check if left node is a leaf

#### Approach

Let's try to reuse the original method's signature.
* in Rust `Rc::clone` is a cheap operation

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(log(n))$$, for the recursion stack space

#### Code

```kotlin

    fun sumOfLeftLeaves(root: TreeNode?): Int = root?.run {
       (left?.takeIf { it.left == null && it.right == null }?.`val` ?:
       sumOfLeftLeaves(left)) + sumOfLeftLeaves(right)
    } ?: 0

```
```rust

    pub fn sum_of_left_leaves(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        root.as_ref().map_or(0, |n| { let n = n.borrow();
            n.left.as_ref().map_or(0, |left| { let l = left.borrow();
                if l.left.is_none() && l.right.is_none() { l.val }
                else { Self::sum_of_left_leaves(Some(Rc::clone(left))) }
            }) +
            n.right.as_ref().map_or(0, |r| Self::sum_of_left_leaves(Some(Rc::clone(r))))
        })
    }

```

