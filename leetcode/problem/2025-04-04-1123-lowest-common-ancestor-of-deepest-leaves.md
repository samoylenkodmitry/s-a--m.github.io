---
layout: leetcode-entry
title: "1123. Lowest Common Ancestor of Deepest Leaves"
permalink: "/leetcode/problem/2025-04-04-1123-lowest-common-ancestor-of-deepest-leaves/"
leetcode_ui: true
entry_slug: "2025-04-04-1123-lowest-common-ancestor-of-deepest-leaves"
---

[1123. Lowest Common Ancestor of Deepest Leaves](https://leetcode.com/problems/lowest-common-ancestor-of-deepest-leaves/description) medium
[blog post](https://leetcode.com/problems/lowest-common-ancestor-of-deepest-leaves/solutions/6613757/kotlin-rust-by-samoylenkodmitry-glkw/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/04042025-1123-lowest-common-ancestor?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/DorteDNZJEs)
![1.webp](/assets/leetcode_daily_images/d5906f3a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/948

#### Problem TLDR

Lowest common ancestor of deepest tree nodes #medium #recursion

#### Intuition

Solve the problem for the left and right subtrees. Update ancestor of both left and right have equal depths.

#### Approach

* provide examples of several deepest nodes to better understand what is asked

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(log(n))$$ recursion depth

#### Code

```kotlin

    fun lcaDeepestLeaves(root: TreeNode?): TreeNode? {
        fun d(n: TreeNode?): Pair<Int, TreeNode?> = n?.run {
            val (a, l) = d(left); val (b, r) = d(right)
            if (a == b) a + 1 to n else if (a > b) a + 1 to l else b + 1 to r
        } ?: 0 to null
        return d(root).second
    }

```
```kotlin

    fun lcaDeepestLeaves(root: TreeNode?): TreeNode? {
        var max = 0; var res = root
        fun d(n: TreeNode?, lvl: Int): Int {
            max = max(lvl, max); n ?: return lvl
            val l = d(n.left, lvl + 1); val r = d(n.right, lvl + 1)
            if (l == max && l == r) res = n
            return max(l, r)
        }
        d(root, 0)
        return res
    }

```
```rust

    pub fn lca_deepest_leaves(root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
        fn d(no: Option<Rc<RefCell<TreeNode>>>) -> (i32, Option<Rc<RefCell<TreeNode>>>) {
            let Some(nrc) = no.clone() else { return (0, no) }; let n = nrc.borrow();
            let (a, l) = d(n.left.clone()); let (b, r) = d(n.right.clone());
            if a == b { (a + 1, no) } else if a > b { (a + 1, l) } else { (b + 1, r) }
        }
        d(root.clone()).1
    }

```
```c++

    TreeNode* lcaDeepestLeaves(TreeNode* root) {
        auto d = [&](this const auto& d, TreeNode* n) -> pair<int, TreeNode*> {
            if (!n) return {0, n}; auto [a, l] = d(n->left); auto [b, r] = d(n->right);
            if (a == b) return {a + 1, n}; if (a > b) return {a + 1, l}; return {b + 1, r};
        };
        return d(root).second;
    }

```

