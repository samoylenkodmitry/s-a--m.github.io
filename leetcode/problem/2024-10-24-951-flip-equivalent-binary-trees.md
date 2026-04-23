---
layout: leetcode-entry
title: "951. Flip Equivalent Binary Trees"
permalink: "/leetcode/problem/2024-10-24-951-flip-equivalent-binary-trees/"
leetcode_ui: true
entry_slug: "2024-10-24-951-flip-equivalent-binary-trees"
---

[951. Flip Equivalent Binary Trees](https://leetcode.com/problems/flip-equivalent-binary-trees/description/) medium
[blog post](https://leetcode.com/problems/flip-equivalent-binary-trees/solutions/5961232/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24102024-951-flip-equivalent-binary?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/EMLccalWwoM)
[deep-dive](https://notebooklm.google.com/notebook/5ec499a5-88e0-414a-ac51-0966e590ca7b/audio)
![1.webp](/assets/leetcode_daily_images/4aaee878.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/778

#### Problem TLDR

Are trees flip-equal #medium #recursion #dfs

#### Intuition

The problem size is small, 100 elements, we can do a full Depth-First Search and emulate swaps

#### Approach

* this problem is a one-liner recursion golf

#### Complexity

- Time complexity:
$$O(n^2)$$, `d = log(n)` recursion depth, each time we try at most `4` searches, so it is `4^d = 4^log(n)`, simplified with identity of $$a^{\log(c)} = c^{\log(a)}$$ to $$
4^{log(n)} = n^{log(4)} = n^{2log_2(2)} = n^2$$

- Space complexity:
$$O(log(n))$$

#### Code

```kotlin

    fun flipEquiv(root1: TreeNode?, root2: TreeNode?): Boolean =
      root1?.run {
        `val` == (root2?.`val` ?: -1) && (
        flipEquiv(left, root2!!.left) &&
        flipEquiv(right, root2.right) ||
        flipEquiv(left, root2.right) &&
        flipEquiv(right, root2.left)) } ?: (root2 == null)

```
```rust

    pub fn flip_equiv(root1: Option<Rc<RefCell<TreeNode>>>,
                      root2: Option<Rc<RefCell<TreeNode>>>) -> bool {
        let Some(r1) = root1 else { return root2.is_none() };
        let Some(r2) = root2 else { return false };
        let (r1, r2) = (r1.borrow(), r2.borrow());
        r1.val == r2.val && (
            Self::flip_equiv(r1.left.clone(), r2.left.clone()) &&
            Self::flip_equiv(r1.right.clone(), r2.right.clone()) ||
            Self::flip_equiv(r1.left.clone(), r2.right.clone()) &&
            Self::flip_equiv(r1.right.clone(), r2.left.clone()))
    }

```
```c++

    bool flipEquiv(TreeNode* root1, TreeNode* root2) {
        return !root1 == !root2 && (!root1 || root1->val == root2->val && (
            flipEquiv(root1->left, root2->left) && flipEquiv(root1->right, root2->right) ||
            flipEquiv(root1->left, root2->right) && flipEquiv(root1->right, root2->left)));
    }

```

