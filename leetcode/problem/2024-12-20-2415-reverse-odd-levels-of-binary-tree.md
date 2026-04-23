---
layout: leetcode-entry
title: "2415. Reverse Odd Levels of Binary Tree"
permalink: "/leetcode/problem/2024-12-20-2415-reverse-odd-levels-of-binary-tree/"
leetcode_ui: true
entry_slug: "2024-12-20-2415-reverse-odd-levels-of-binary-tree"
---

[2415. Reverse Odd Levels of Binary Tree](https://leetcode.com/problems/reverse-odd-levels-of-binary-tree/description/) medium
[blog post](https://leetcode.com/problems/reverse-odd-levels-of-binary-tree/solutions/6167057/kotlin-rust-by-samoylenkodmitry-1koo/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20122024-2415-reverse-odd-levels?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/5bS6Z43R5v0)
[deep-dive](https://notebooklm.google.com/notebook/19d69569-582b-4b01-987d-86c4c1fd6705/audio)
![1.webp](/assets/leetcode_daily_images/36a0f567.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/838

#### Problem TLDR

Odd-levels reversal in a perfect tree #medium #dfs #bfs #tree

#### Intuition

The most straightforward way is a level-order Breadth-First Search traversal. Remember the previous layer, adjust current values accordingly.

The more interesting way is how you can do it recursively:
* pass outer-left and outer-right values `a` and `b` (and a depth)
* pass inner-right and inner-left values as `a` and `b`
* swapping `a` and `b` *will* result in the all level reversal (that's an interesting fact that should be observed and discovered until understood)

#### Approach

* let's implement both BFS and DFS
* remember to reverse only `odd` levels
* rewrite the `values` instead of the pointers, much simpler code

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$ or O(log(n)) for recursion

#### Code

```kotlin

    fun reverseOddLevels(root: TreeNode?): TreeNode? {
        fun f(l: TreeNode?, r: TreeNode?, d: Int) {
            l ?: return; r ?: return
            if (d % 2 > 0) l.`val` = r.`val`.also { r.`val` = l.`val` }
            f(l.left, r.right, d + 1)
            f(l.right, r.left, d + 1)
        }
        f(root?.left, root?.right, 1)
        return root
    }

```
```rust

    pub fn reverse_odd_levels(root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
        let Some(r) = root.clone() else { return root };  let mut q = VecDeque::from([r]);
        let (mut i, mut vs) = (0, vec![]);
        while q.len() > 0 {
            let mut l = vec![];
            for _ in 0..q.len() {
                let n = q.pop_front().unwrap(); let mut n = n.borrow_mut();
                if i % 2 > 0 && vs.len() > 0 { n.val = vs.pop().unwrap(); }
                if let Some(x) = n.left.clone() { l.push(x.borrow().val); q.push_back(x); }
                if let Some(x) = n.right.clone() { l.push(x.borrow().val); q.push_back(x); }
            }
            vs = l; i += 1
        }
        root
    }

```
```c++

    TreeNode* reverseOddLevels(TreeNode* root) {
        auto f = [](this auto const& f, TreeNode* l, TreeNode* r, int d) {
            if (!l || !r) return; if (d % 2) swap(l->val, r->val);
            f(l->left, r->right, d + 1); f(l->right, r->left, d + 1);
        };
        f(root->left, root->right, 1); return root;
    }

```

